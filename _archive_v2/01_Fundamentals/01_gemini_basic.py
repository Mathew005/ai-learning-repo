import os
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

async def main():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in .env")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    print("--- Connecting to Gemini ---")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Explain Quantum Computing in 1 sentence."
        )
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
