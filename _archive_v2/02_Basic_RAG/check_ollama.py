import requests
import sys

def check_ollama():
    url = "http://localhost:11434/api/embeddings"
    model = "nomic-embed-text"
    
    print(f"🔍 Checking Ollama connection at {url}...")
    
    payload = {
        "model": model,
        "prompt": "Hello world"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"✅ Ollama is running and model '{model}' is ready.")
            print("Sample vector prefix:", response.json().get("embedding")[:5])
        elif response.status_code == 404:
             print(f"❌ Ollama Connected, but model '{model}' not found.")
             print(f"👉 Please run: ollama pull {model}")
             sys.exit(1)
        else:
            print(f"❌ Error talking to Ollama: {response.status_code} - {response.text}")
            sys.exit(1)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama. Is it running?")
        print("👉 Run 'ollama serve' in a separate terminal.")
        sys.exit(1)

if __name__ == "__main__":
    check_ollama()
