import os, requests, json
from dotenv import load_dotenv
from pathlib import Path

# Bug fix: .env is in the SAME directory as this script, not two levels up
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"

if not key:
    print("ERROR: OPENROUTER_API_KEY not found in .env — check that your .env file exists")
    print(f"       and contains OPENROUTER_API_KEY=<your-key>")
    print(f"       Looked in: {env_path}")
    exit(1)

headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

# Bug fix: use the SAME model as config.py, and test JSON mode exactly as agent.py does
from config import OPENROUTER_MODEL
model = OPENROUTER_MODEL

# Bug fix: include response_format so we test the real agent code path (JSON mode)
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You output JSON only."},
        {"role": "user", "content": 'Reply with {"status": "ok"}'},
    ],
    "response_format": {"type": "json_object"},
    "temperature": 0,
}

print(f"Testing URL: {url}")
print(f"Testing Model: {model}")
print(f"API Key prefix: {key[:10]}...")

try:
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status Code: {r.status_code}")
    if r.status_code == 200:
        content = r.json()["choices"][0]["message"]["content"]
        print(f"Raw response: {content}")
        parsed = json.loads(content)
        print(f"Parsed JSON OK: {parsed}")
        print("\n✓ OpenRouter + JSON mode working correctly!")
    else:
        print(f"Error response: {r.text}")
except json.JSONDecodeError as e:
    print(f"JSON parse failed: {e} — model may not support json_object mode")
except Exception as e:
    print(f"Request error: {e}")
