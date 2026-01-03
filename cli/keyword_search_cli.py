#!/usr/bin/env python3

import argparse, json
from search_function import search_function
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_function(args.query)
            for key,value in enumerate(results[:5], 1):
                print(key, value["title"])
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()