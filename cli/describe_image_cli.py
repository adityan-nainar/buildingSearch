import argparse
import mimetypes
from lib.hybrid_search import rrf_search_command
from google.genai import types

from dotenv import load_dotenv
from google import genai
import os

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"

def main():
    parser = argparse.ArgumentParser(description="Rewrite a query based on an image using Gemini.")
    parser.add_argument("--image", required=True, help="Path to the image file (e.g., data/paddington.jpeg)")
    parser.add_argument("--query", required=True, help="Text query to rewrite")
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    try:
        with open(args.image, "rb") as f:
            img_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: The file {args.image} was not found.")
        return

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash-001"

    system_prompt = (
        "Given the included image and text query, rewrite the text query to improve "
        "search results from a movie database. Make sure to:\n"
        "- Synthesize visual and textual information\n"
        "- Focus on movie-specific details (actors, scenes, style, etc.)\n"
        "- Return only the rewritten query, without any additional commentary"
    )

    parts = [
        system_prompt,
        types.Part.from_bytes(data=img_bytes, mime_type=mime),
        args.query.strip(),
    ]

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=parts
        )

        if response.text:
            print(f"Rewritten query: {response.text.strip()}")
        
        if response.usage_metadata is not None:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")
            
    except Exception as e:
        print(f"An error occurred during generation: {e}")


if __name__ == "__main__":
    main()