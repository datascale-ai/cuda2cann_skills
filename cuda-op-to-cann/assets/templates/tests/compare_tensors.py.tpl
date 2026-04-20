#!/usr/bin/env python3
"""Numerical comparison template for migrated operators."""

import torch


def reference_impl(x, y):
    raise NotImplementedError


def migrated_impl(x, y):
    raise NotImplementedError


def main():
    x = torch.randn(2, 3, dtype=torch.float16)
    y = torch.randn(2, 3, dtype=torch.float16)
    torch.testing.assert_close(migrated_impl(x, y), reference_impl(x, y), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    main()
