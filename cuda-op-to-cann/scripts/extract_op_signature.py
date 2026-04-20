#!/usr/bin/env python3
"""Extract a best-effort operator signature from a CUDA project."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import (
    camel_case,
    discover_build_files,
    discover_sources,
    infer_scalar_type,
    infer_tensor_type,
    project_root_for_files,
    snake_case,
    split_arguments,
)

CONTROL_KEYWORDS = {"if", "for", "while", "switch", "catch"}
FUNCTION_RE = re.compile(
    r"(?P<prefix>(?:template\s*<[^{};]+>\s*)?(?:[\w:&*<>\[\]\s\"']+?)?)"
    r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{}]*)\)\s*\{",
    re.S,
)
KERNEL_RE = re.compile(
    r"__global__\s+(?:__launch_bounds__\([^)]*\)\s*)?(?:[\w:&*<>\[\]\s]+?)"
    r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)",
    re.S,
)
LAUNCH_RE = re.compile(r"(?P<kernel>[A-Za-z_]\w*)\s*<<<(?P<cfg>.*?)>>>\s*\((?P<args>.*?)\)\s*;", re.S)
REGISTRATION_RE = re.compile(
    r'm\.(?:def|impl)\(\s*"(?P<op>[^"]+)"(?:\s*,\s*&?(?P<target>[A-Za-z_]\w*))?',
    re.S,
)


def strip_comments(content: str) -> str:
    content = re.sub(r"//.*?$", "", content, flags=re.M)
    return re.sub(r"/\*.*?\*/", "", content, flags=re.S)


def find_matching_brace(content: str, brace_index: int) -> int:
    depth = 0
    for index in range(brace_index, len(content)):
        char = content[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(content) - 1


def parse_param(param: str) -> dict | None:
    raw = param.strip()
    if not raw or raw == "void":
        return None
    raw = raw.split("=")[0].strip()
    match = re.match(r"(?P<type>.+?)\s*(?P<name>[*&]*[A-Za-z_]\w*)$", raw)
    if not match:
        return {
            "name": snake_case(raw)[:24] or "arg",
            "cpp_type": raw,
            "kind": "scalar",
            "role": "attr",
            "attr_type": "int",
        }
    cpp_type = match.group("type").strip()
    name = match.group("name").strip()
    while name and name[0] in "*&":
        cpp_type = f"{cpp_type}{name[0]}"
        name = name[1:]
    lower_type = cpp_type.lower()
    lower_name = name.lower()

    if "stream" in lower_name or "cudaStream" in cpp_type:
        kind = "stream"
        role = "meta"
    elif "workspace" in lower_name:
        kind = "workspace"
        role = "meta"
    elif "tensor" in lower_type or re.search(r"\b(?:half|float|double|int\d*|bool)\s*[*&]", lower_type):
        kind = "tensor"
        if "const" in lower_type:
            role = "input"
        elif re.search(r"(out|output|result|dst|grad|z)$", lower_name):
            role = "output"
        else:
            role = "input"
    else:
        kind = "scalar"
        role = "attr"

    parsed = {
        "name": name,
        "cpp_type": cpp_type,
        "kind": kind,
        "role": role,
    }
    if kind == "tensor":
        parsed["supported_types"] = infer_tensor_type(cpp_type)
    elif kind == "scalar":
        parsed["attr_type"] = infer_scalar_type(cpp_type)
    return parsed


def extract_functions(content: str) -> list[dict]:
    functions: list[dict] = []
    cleaned = strip_comments(content)
    for match in FUNCTION_RE.finditer(cleaned):
        name = match.group("name")
        if name in CONTROL_KEYWORDS:
            continue
        prefix = " ".join(match.group("prefix").split())
        open_brace = cleaned.find("{", match.end() - 1)
        close_brace = find_matching_brace(cleaned, open_brace)
        params = [param for param in (parse_param(item) for item in split_arguments(match.group("params"))) if param]
        functions.append(
            {
                "name": name,
                "return_type": prefix.strip(),
                "params": params,
                "start": match.start(),
                "end": close_brace,
                "body": cleaned[open_brace:close_brace + 1],
            }
        )
    return functions


def extract_kernels(content: str) -> list[dict]:
    kernels = []
    cleaned = strip_comments(content)
    for match in KERNEL_RE.finditer(cleaned):
        params = [param for param in (parse_param(item) for item in split_arguments(match.group("params"))) if param]
        kernels.append({"name": match.group("name"), "params": params})
    return kernels


def collect_registrations(content: str) -> list[dict]:
    return [match.groupdict() for match in REGISTRATION_RE.finditer(strip_comments(content))]


def build_primary_signature(functions: list[dict], kernels: list[dict], registrations: list[dict]) -> dict:
    registered_targets = {item["target"] for item in registrations if item.get("target")}
    candidate = None
    for function in functions:
        if function["name"] in registered_targets:
            candidate = function
            break
    if candidate is None:
        for function in functions:
            if LAUNCH_RE.search(function["body"]):
                candidate = function
                break
    if candidate is None and functions:
        candidate = functions[0]

    raw_name = registrations[0]["op"] if registrations else (candidate["name"] if candidate else (kernels[0]["name"] if kernels else "custom_op"))
    op_type = raw_name if re.search(r"[a-z][A-Z]", raw_name) and not re.search(r"[_\-\s]+", raw_name) else camel_case(raw_name)
    snake_name = snake_case(raw_name)
    params = candidate["params"] if candidate else []

    inputs = [item for item in params if item["role"] == "input"]
    outputs = [item for item in params if item["role"] == "output"]
    attrs = [item for item in params if item["role"] == "attr"]
    if not outputs and candidate and "tensor" in candidate["return_type"].lower():
        outputs = [
            {
                "name": "out",
                "cpp_type": candidate["return_type"],
                "kind": "tensor",
                "role": "output",
                "supported_types": infer_tensor_type(candidate["return_type"]),
            }
        ]

    linked_kernel = None
    launch_args = []
    if candidate:
        launch_match = LAUNCH_RE.search(candidate["body"])
        if launch_match:
            linked_kernel = launch_match.group("kernel")
            launch_args = [item.strip() for item in split_arguments(launch_match.group("args")) if item.strip()]

    kernel = None
    if linked_kernel:
        kernel = next((item for item in kernels if item["name"] == linked_kernel), None)
    if kernel is None and kernels:
        kernel = kernels[0]

    return {
        "op_name": raw_name,
        "op_type": op_type,
        "op_name_snake": snake_name,
        "wrapper_function": candidate["name"] if candidate else None,
        "kernel_function": kernel["name"] if kernel else None,
        "return_type": candidate["return_type"] if candidate else "void",
        "inputs": inputs,
        "outputs": outputs,
        "attrs": attrs,
        "launch_args": launch_args,
        "registered_ops": [item["op"] for item in registrations],
    }


def extract_project_signature(src_paths: list[str]) -> dict:
    files = discover_sources(src_paths)
    functions: list[dict] = []
    kernels: list[dict] = []
    registrations: list[dict] = []
    for file_path in files:
        content = file_path.read_text(errors="ignore")
        functions.extend(extract_functions(content))
        kernels.extend(extract_kernels(content))
        registrations.extend(collect_registrations(content))

    project_root = project_root_for_files(files)
    build_files = discover_build_files(files)
    primary = build_primary_signature(functions, kernels, registrations)
    return {
        "project_root": str(project_root) if project_root else None,
        "build_files": [str(path) for path in build_files],
        "functions": [
            {
                "name": item["name"],
                "return_type": item["return_type"],
                "params": item["params"],
            }
            for item in functions[:20]
        ],
        "kernels": kernels,
        "registrations": registrations,
        "primary_signature": primary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", nargs="+", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = extract_project_signature(args.src)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
