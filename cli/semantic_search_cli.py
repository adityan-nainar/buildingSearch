#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch
from lib.search_utils import load_movies, format_search_result

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    embed_parser =subparsers.add_parser("embed_text", help="Embed text")
    embed_parser.add_argument("query", type=str, help="Text for embedding")

    verify_embed_parser = subparsers.add_parser("verify_embeddings", help="Verify Embeddings")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query")
    embed_query_parser.add_argument("query", type=str, help="Text for embedding")

    search_parser = subparsers.add_parser("search", help="Embed query")
    search_parser.add_argument("query", type=str, help="Text for embedding")
    search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        
        case "embed_text":
            embed_text(args.query)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":  
            search_instance = SemanticSearch()
            search_instance.load_or_create_embeddings(load_movies())
            results = search_instance.search(args.query, args.limit)

            # results is a list of (score, movie_dict)
            for i, (score, doc) in enumerate(results, 1):
                # Format: 1. Title (score: 0.####)
                print(f"{i}. {doc['title']} (score: {score:.4f})")
                # Print the description on the next line with indentation
                print(f"   {doc['description']}")
                print() # Adds a blank line between results

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
