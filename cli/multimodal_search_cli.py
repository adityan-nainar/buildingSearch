import argparse
import sys
from lib.multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI Tools")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding")
    verify_parser.add_argument("image_path", type=str)

    search_parser = subparsers.add_parser("image_search")
    search_parser.add_argument("image_path", type=str)

    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)
    
    elif args.command == "image_search":
        results = image_search_command(args.image_path)
        
        for i, res in enumerate(results, 1):
            truncated_score = int(res['score'] * 1000) / 1000
            
            print(f"{i}. {res['title']} (similarity: {truncated_score:.3f})")
            
            desc = res['description']
            print(f"   {desc[:100]}...\n")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()