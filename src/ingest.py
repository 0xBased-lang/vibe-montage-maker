import os
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import yt_dlp
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import chromadb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

COOKIES_FILE = os.environ.get("COOKIES_FILE")  # optional: for IG/TikTok gated content

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


class VibeEngine:
    """Handles embedding of frames into the semantic vector DB."""

    def __init__(self):
        print("Loading SigLIP model (google/siglip-base-patch16-224)...")
        self.model_name = "google/siglip-base-patch16-224"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        print("Initializing ChromaDB store...")
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name="video_frames")

    def embed_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return embeddings.squeeze().tolist()
        except Exception as exc:
            print(f"[WARN] Failed to embed {image_path}: {exc}")
            return None

    def add_frame(self, frame_path, video_id, timestamp):
        vector = self.embed_image(frame_path)
        if vector is None:
            return
        self.collection.add(
            embeddings=[vector],
            documents=[frame_path],
            metadatas=[{"video_id": video_id, "timestamp": timestamp}],
            ids=[f"{video_id}_{timestamp}"]
        )
        print(f"Indexed \"{os.path.basename(frame_path)}\"")


def download_video(url):
    print(f"Downloading: {url}")
    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "noplaylist": True,
    }
    if COOKIES_FILE:
        ydl_opts["cookiefile"] = COOKIES_FILE
        print(f"Using cookies file: {COOKIES_FILE}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return os.path.join(DOWNLOAD_DIR, f"{info['id']}.{info['ext']}"), info["id"]


def extract_scene_frames(video_path, video_id):
    print(f"Detecting scenes in {os.path.basename(video_path)}")
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    print(f"Found {len(scenes)} scenes")

    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx, scene in enumerate(scenes):
        start, end = scene
        middle_frame = start.get_frames() + (end.get_frames() - start.get_frames()) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        success, frame = cap.read()
        if not success:
            continue
        timestamp = start.get_seconds()
        filename = f"{video_id}_scene_{idx}_{int(timestamp)}.jpg"
        save_path = os.path.join(FRAMES_DIR, filename)
        cv2.imwrite(save_path, frame)
        frames.append((save_path, timestamp))
    cap.release()
    video_manager.release()
    return frames


def main():
    engine = VibeEngine()
    url = input("Enter video URL (YouTube/TikTok/Instagram): ").strip()
    if not url:
        print("No URL provided. Exiting.")
        return
    try:
        video_path, video_id = download_video(url)
        frames = extract_scene_frames(video_path, video_id)
        print(f"Embedding {len(frames)} frames...")
        for frame_path, timestamp in frames:
            engine.add_frame(frame_path, video_id, timestamp)
        print("Ingestion complete. Frames ready for searching.")
    except Exception as exc:
        print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
