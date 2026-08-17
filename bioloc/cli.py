"""Small command-line entry point for package inspection."""

from __future__ import annotations

import argparse
from importlib.metadata import version

from .config import config_help
from .models import output_help


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bioloc")
    parser.add_argument("--version", action="store_true")
    parser.add_argument(
        "--config-help",
        nargs="?",
        const="all",
        metavar="SECTION",
        help="show all pipeline parameters, or only one configuration section",
    )
    parser.add_argument(
        "--output-help",
        nargs="?",
        const="all",
        metavar="SECTION",
        help="describe all pipeline outputs, or only one result section",
    )
    args = parser.parse_args(argv)
    if args.version:
        print(version("bioloc"))
        return 0
    if args.config_help is not None:
        try:
            output = config_help(args.config_help)
        except ValueError as error:
            parser.error(str(error))
        print(output)
        return 0
    if args.output_help is not None:
        try:
            output = output_help(args.output_help)
        except ValueError as error:
            parser.error(str(error))
        print(output)
        return 0
    parser.print_help()
    return 0
