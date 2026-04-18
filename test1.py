import requests
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Testiraj oba URL-a
urls = [
    "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
    "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
]

headers = {"Authorization": f"Bearer {HF_TOKEN}"}
payload = {"inputs": ["test rečenica"], "options": {"wait_for_model": True}}

for url in urls:
    print(f"\n--- Testiram: {url} ---")
    r = requests.post(url, headers=headers, json=payload)
    print(f"Status: {r.status_code}")
    print(f"Odgovor: {r.text[:300]}")