import asyncio
import httpx

OLLAMA_BASE_URL = "http://localhost:11434"

async def main():
    print("--- Connecting to Ollama ---")
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": "gemma:2b", # Change model as needed
        "prompt": "Explain Quantum Computing in 1 sentence.",
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=60.0)
            if resp.status_code == 200:
                print(f"Response: {resp.json().get('response')}")
            else:
                print(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
