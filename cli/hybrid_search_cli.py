import argparse
from lib.hybrid_search import hybrid_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help = "normalize vectors")
    normalize_parser.add_argument("scores", type = float, nargs="+")

    weighted_parser = subparsers.add_parser("weighted-search", help = "weighted search")
    weighted_parser.add_argument("query", type=str, help = "query for searching")
    weighted_parser.add_argument("--alpha", type=float, help = "weight of the hybrid search, 1 is 100% keyword, 0 is 100% Semantic")
    weighted_parser.add_argument("--limit", type=int, help = "query for searching")

    rrf_parser = subparsers.add_parser("rrf-search", help="RRF hybrid search")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF constant k")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results")


    args = parser.parse_args()
    match args.command:
        case "normalize":
            num_list = args.scores
            minn = min(num_list)
            maxx = max(num_list)
            if minn==maxx:
                for i in range(len(num_list)):
                    print(1.0)
            else:
                for score in num_list:
                    new_score = (score-minn)/(maxx-minn)
                    print(f"* {new_score:.4f}")

        case "weighted-search":
            results = hybrid_search(args.query, args.alpha, args.limit)
            # Format the output as requested
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res['hybrid_score']:.3f}")
                print(f"   BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                print(f"   {res['description'][:100]}...") # Truncate description for display

        case "rrf-search":
            from lib.hybrid_search import rrf_search_helper
            results = rrf_search_helper(args.query, args.k, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f"   RRF Score: {res['rrf_score']:.4f}")
                bm25_rank = res['bm25_rank'] if res['bm25_rank'] else "N/A"
                semantic_rank = res['semantic_rank'] if res['semantic_rank'] else "N/A"
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"   {res['description'][:100]}...")
                
        case _: 
            parser.print_help()


if __name__ == "__main__":
    main()