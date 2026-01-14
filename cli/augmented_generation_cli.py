import argparse

from lib.hybrid_search import rrf_search_command

from dotenv import load_dotenv
from google import genai
import os

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Perform Summarize RAG (search + generate answer)")
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
     
    citations_parser = subparsers.add_parser("citations", help="Perform Citations RAG (search + generate answer)")
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument("--limit", type=int, help="limit")

    question_parser = subparsers.add_parser("question", help="Perform Citations RAG (search + generate answer)")
    question_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            result = rrf_search_command(args.query)
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                if "individual_score" in res:
                    print(f"   Rerank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Rerank Rank: {res.get('batch_rank', 0)}")
                if "crossencoder_score" in res:
                    print(
                        f"   Cross Encoder Score: {res.get('crossencoder_score', 0):.3f}"
                    )
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()
            
            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Documents:
            {result}

            Provide a comprehensive answer that addresses the query:"""

            response = client.models.generate_content(model=model, contents=prompt)
            print(response.text)

        case "summarize":
            query = args.query
            result = rrf_search_command(args.query)
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                if "individual_score" in res:
                    print(f"   Rerank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Rerank Rank: {res.get('batch_rank', 0)}")
                if "crossencoder_score" in res:
                    print(
                        f"   Cross Encoder Score: {res.get('crossencoder_score', 0):.3f}"
                    )
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()
            
            prompt = f"""
            Provide information useful to this query by synthesizing information from multiple search results in detail.
            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
            This should be tailored to Hoopla users. Hoopla is a movie streaming service.
            Query: {query}
            Search Results:
            {result}
            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
            """


            response = client.models.generate_content(model=model, contents=prompt)
            print(response.text)

        case "citations":
            query = args.query
            result = rrf_search_command(args.query)
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                if "individual_score" in res:
                    print(f"   Rerank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Rerank Rank: {res.get('batch_rank', 0)}")
                if "crossencoder_score" in res:
                    print(
                        f"   Cross Encoder Score: {res.get('crossencoder_score', 0):.3f}"
                    )
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()
            
            prompt = f"""Answer the question or provide information based on the provided documents.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

                Query: {query}

                Documents:
                {result}

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources using [1], [2], etc. format when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the documents, say "I don't have enough information"
                - Be direct and informative

                Answer:"""


            response = client.models.generate_content(model=model, contents=prompt)
            print(response.text)
        
        case "question":
            query = args.query
            result = rrf_search_command(args.query)
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                if "individual_score" in res:
                    print(f"   Rerank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Rerank Rank: {res.get('batch_rank', 0)}")
                if "crossencoder_score" in res:
                    print(
                        f"   Cross Encoder Score: {res.get('crossencoder_score', 0):.3f}"
                    )
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()
            
            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

                    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                    Question: {query}

                    Documents:
                    {result}

                    Instructions:
                    - Answer questions directly and concisely
                    - Be casual and conversational
                    - Don't be cringe or hype-y
                    - Talk like a normal person would in a chat conversation

                    Answer:"""


            response = client.models.generate_content(model=model, contents=prompt)
            print(response.text)
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()