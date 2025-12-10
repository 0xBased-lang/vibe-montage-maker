import os
import shutil
import chromadb
from transformers import AutoTokenizer, AutoModel
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
EXPORT_DIR = os.path.join(BASE_DIR, "organized_screenshots")

os.makedirs(EXPORT_DIR, exist_ok=True)

class VibeSearch:
    def __init__(self):
        print("Loading SigLIP model for text search...")
        self.model_name = "google/siglip-base-patch16-224"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_collection(name="video_frames")

    def embed_text(self, query):
        inputs = self.tokenizer([query], padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return embedding.squeeze().tolist()

    def search(self, query, limit):
        vector = self.embed_text(query)
        results = self.collection.query(query_embeddings=[vector], n_results=limit)
        return results

    def export(self, query, limit):
        results = self.search(query, limit)
        docs = results.get("documents", [[]])[0]
        if not docs:
            print("No matches found.")
            return
        distances = results.get("distances", [[]])[0]
        safe_dir = query.strip().replace(" ", "_").lower() or "untitled"
        target_dir = os.path.join(EXPORT_DIR, safe_dir)
        os.makedirs(target_dir, exist_ok=True)

        for idx, (doc, distance) in enumerate(zip(docs, distances), start=1):
            if not os.path.exists(doc):
                print(f"[WARN] Missing file: {doc}")
                continue
            new_name = f"{idx:02d}_{os.path.basename(doc)}"
            shutil.copy2(doc, os.path.join(target_dir, new_name))
            print(f"[{idx}] {new_name} (score={distance:.4f})")
        print(f"Exported to {target_dir}")


def main():
    searcher = VibeSearch()
    while True:
        query = input("Describe desired vibe (or 'q' to quit): ").strip()
        if not query or query.lower() == "q":
            break
        try:
            limit = int(input("How many screenshots? [10]: ") or 10)
        except ValueError:
            limit = 10
        searcher.export(query, limit)

if __name__ == "__main__":
    main()
