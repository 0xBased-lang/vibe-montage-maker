import csv
import json
import os
import shutil
import wave

import chromadb
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
EXPORT_DIR = os.path.join(BASE_DIR, "organized_screenshots")
CAPCUT_DIR = os.path.join(BASE_DIR, "capcut_ready")
DEFAULT_BPM = 120

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(CAPCUT_DIR, exist_ok=True)


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

    def export(self, query, limit, bpm=DEFAULT_BPM):
        results = self.search(query, limit)
        docs = results.get("documents", [[]])[0]
        if not docs:
            print("No matches found.")
            return

        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        safe_dir = query.strip().replace(" ", "_").lower() or "untitled"
        vibe_dir = os.path.join(EXPORT_DIR, safe_dir)
        capcut_dir = os.path.join(CAPCUT_DIR, safe_dir)
        os.makedirs(vibe_dir, exist_ok=True)
        os.makedirs(capcut_dir, exist_ok=True)

        export_entries = []

        for idx, (doc, distance, meta) in enumerate(
            zip(docs, distances, metadatas), start=1
        ):
            if not os.path.exists(doc):
                print(f"[WARN] Missing file: {doc}")
                continue

            ranked_name = f"{idx:02d}_{os.path.basename(doc)}"
            shutil.copy2(doc, os.path.join(vibe_dir, ranked_name))

            capcut_name = f"{idx:04d}.jpg"
            shutil.copy2(doc, os.path.join(capcut_dir, capcut_name))

            export_entries.append(
                {
                    "rank": idx,
                    "ranked_name": ranked_name,
                    "capcut_name": capcut_name,
                    "original_path": doc,
                    "distance": float(distance),
                    "metadata": meta or {},
                }
            )

            print(f"[{idx}] {ranked_name} (score={distance:.4f})")

        if not export_entries:
            print("No files copied. Aborting extras.")
            return

        self._write_shot_metadata(capcut_dir, export_entries)
        beat_duration = self._generate_click_track(
            capcut_dir, len(export_entries), bpm=bpm
        )
        self._write_manifest(capcut_dir, query, export_entries, bpm, beat_duration)
        self._write_script_stub(capcut_dir, query, export_entries)
        self._zip_capcut_folder(capcut_dir, safe_dir)

        print(f"\nExported to {vibe_dir}")
        print(f"CapCut assets ready in {capcut_dir}")
        print(
            "Shots metadata, manifest, click track, and archive generated for CapCut."
        )

    def _write_shot_metadata(self, capcut_dir, entries):
        csv_path = os.path.join(capcut_dir, "shots.csv")
        headers = [
            "rank",
            "capcut_file",
            "ranked_file",
            "source_video",
            "source_timestamp",
            "distance",
            "original_path",
        ]
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            for entry in entries:
                meta = entry["metadata"]
                writer.writerow(
                    [
                        entry["rank"],
                        entry["capcut_name"],
                        entry["ranked_name"],
                        meta.get("video_id", "unknown"),
                        meta.get("timestamp", "unknown"),
                        f"{entry['distance']:.4f}",
                        entry["original_path"],
                    ]
                )

    def _generate_click_track(self, capcut_dir, beats, bpm=DEFAULT_BPM, sample_rate=44100):
        if beats <= 0:
            return 0
        beat_duration = 60.0 / bpm
        click_duration = min(0.05, beat_duration / 2)
        t_click = np.linspace(0, click_duration, int(sample_rate * click_duration), False)
        click = 0.6 * np.sin(2 * np.pi * 1000 * t_click)
        silence = np.zeros(int(sample_rate * (beat_duration - click_duration)))
        pattern = []
        for _ in range(beats):
            pattern.append(np.concatenate([click, silence]))
        audio = np.concatenate(pattern)
        max_val = np.max(np.abs(audio)) or 1
        audio = np.int16(audio / max_val * 32767)
        wav_path = os.path.join(capcut_dir, "click_track.wav")
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return beat_duration

    def _write_manifest(self, capcut_dir, query, entries, bpm, beat_duration):
        manifest = {
            "vibe": query,
            "bpm": bpm,
            "beat_duration_seconds": beat_duration,
            "clips": [],
        }
        for entry in entries:
            meta = entry["metadata"]
            manifest["clips"].append(
                {
                    "rank": entry["rank"],
                    "file": entry["capcut_name"],
                    "suggested_duration": beat_duration,
                    "source_video": meta.get("video_id"),
                    "source_timestamp": meta.get("timestamp"),
                    "similarity_score": entry["distance"],
                }
            )
        manifest_path = os.path.join(capcut_dir, "capcut_manifest.json")
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, indent=2)

    def _write_script_stub(self, capcut_dir, query, entries):
        stub_path = os.path.join(capcut_dir, "script_stub.md")
        lines = [
            f"# Script Notes for \"{query}\"",
            "",
            "Use this template to outline narration or captions for each shot.",
            "",
        ]
        for entry in entries:
            meta = entry["metadata"]
            timestamp = meta.get("timestamp", "unknown")
            lines.append(
                f"- **Shot {entry['rank']}** (`{entry['capcut_name']}`) from t={timestamp}s — notes: "
            )
        with open(stub_path, "w") as stub_file:
            stub_file.write("\n".join(lines))

    def _zip_capcut_folder(self, capcut_dir, safe_dir):
        zip_base = os.path.join(CAPCUT_DIR, f"{safe_dir}_capcut")
        zip_path = f"{zip_base}.zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)
        shutil.make_archive(zip_base, "zip", capcut_dir)


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
        try:
            bpm = int(input(f"Click-track BPM [{DEFAULT_BPM}]: ") or DEFAULT_BPM)
        except ValueError:
            bpm = DEFAULT_BPM
        searcher.export(query, limit, bpm=bpm)


if __name__ == "__main__":
    main()
