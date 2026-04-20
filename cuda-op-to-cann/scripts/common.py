#!/usr/bin/env python3
"""Shared helpers for CUDA-to-CANN migration scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

SOURCE_SUFFIXES = {
    ".cu",
    ".cuh",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".py",
}

BUILD_FILENAMES = {
    "CMakeLists.txt",
    "Makefile",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
}


def camel_case(name: str) -> str:
    if re.search(r"[a-z][A-Z]", name) and not re.search(r"[_\-\s]+", name):
        return name[0].upper() + name[1:]
    return "".join(part.capitalize() for part in re.split(r"[_\-\s]+", name) if part)


def lower_camel_case(name: str) -> str:
    rendered = camel_case(name)
    if not rendered:
        return name
    return rendered[0].lower() + rendered[1:]


def snake_case(name: str) -> str:
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return name.lower().strip("_") or "custom_op"


def normalize_soc_short(soc: str) -> str:
    soc = soc.strip()
    if soc.lower().startswith("ascend") and soc[-1].isdigit():
        return soc[:-1]
    return soc


def soc_candidates(soc: str) -> list[str]:
    raw = soc.strip()
    short = normalize_soc_short(raw)
    values: list[str] = []
    for item in (raw, short):
        if item and item not in values:
            values.append(item)
    lower = short.lower()
    if lower.startswith("ascend910b") or lower.startswith("ascend910a"):
        base = "Ascend910"
        if base not in values:
            values.append(base)
    return values


def discover_sources(src_paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw in src_paths:
        path = Path(raw).expanduser().resolve()
        if path.is_file() and path.suffix in SOURCE_SUFFIXES:
            files.append(path)
            continue
        if path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix in SOURCE_SUFFIXES:
                    files.append(candidate)
    return sorted(set(files))


def project_root_for_files(files: list[Path]) -> Path | None:
    if not files:
        return None
    root = Path(files[0]).parent
    for file_path in files[1:]:
        while not str(file_path).startswith(str(root)):
            if root.parent == root:
                return root
            root = root.parent
    return root


def discover_build_files(files: list[Path]) -> list[Path]:
    root = project_root_for_files(files)
    if root is None:
        return []
    found = []
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.name in BUILD_FILENAMES:
            found.append(candidate)
    return sorted(set(found))


def split_arguments(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth_angle = 0
    depth_round = 0
    depth_square = 0
    for char in text:
        if char == "<":
            depth_angle += 1
        elif char == ">":
            depth_angle = max(0, depth_angle - 1)
        elif char == "(":
            depth_round += 1
        elif char == ")":
            depth_round = max(0, depth_round - 1)
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square = max(0, depth_square - 1)
        elif char == "," and depth_angle == 0 and depth_round == 0 and depth_square == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def infer_scalar_type(cpp_type: str) -> str:
    lower = cpp_type.lower()
    if "bool" in lower:
        return "bool"
    if "int64" in lower or "long" in lower:
        return "int"
    if "int" in lower:
        return "int"
    if "float" in lower or "double" in lower:
        return "float"
    return "int"


def infer_tensor_type(cpp_type: str) -> list[str]:
    lower = cpp_type.lower()
    if "bfloat" in lower:
        return ["bf16", "float"]
    if "half" in lower:
        return ["fp16", "float"]
    if "int64" in lower:
        return ["int64", "int32"]
    if "int" in lower:
        return ["int32"]
    if "double" in lower:
        return ["double", "float"]
    if "bool" in lower:
        return ["bool"]
    return ["fp16", "float", "int32"]


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
