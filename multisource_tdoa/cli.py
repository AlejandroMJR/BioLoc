"""Small command-line entry point for package inspection."""

from __future__ import annotations

import argparse
from importlib.metadata import version


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="multisource-tdoa")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        print(version("multisource-tdoa"))
        return 0
    parser.print_help()
    return 0
