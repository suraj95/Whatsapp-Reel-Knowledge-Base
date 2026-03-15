import base64
import glob
import os
import subprocess
import tempfile
from typing import List

from openai import OpenAI
from yt_dlp import YoutubeDL


def _extract_frames(video_path: str, frames_dir: str) -> List[str]:
    os.makedirs(frames_dir, exist_ok=True)
    # 0.5 fps = 1 frame every 2 seconds
    frame_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        "fps=0.5",
        "-qscale:v",
        "2",
        frame_pattern,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # ffmpeg could not read the file (corrupt / not actually a video)
        return []

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    return frame_paths


def summarize_video_with_gpt4o(client: OpenAI, reel_url: str) -> str:
    """
    Download the reel video, extract frames (1 every 2 seconds) with ffmpeg,
    send a subset of those frames to GPT-4o Vision, and return a short summary.

    Requires ffmpeg to be installed and available on PATH.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "reel.mp4")
        frames_dir = os.path.join(tmpdir, "frames")

        # Download using yt-dlp (supports Instagram, etc.)
        ydl_opts = {
            "outtmpl": video_path,
            "format": "mp4/bestvideo+bestaudio/best",
            "quiet": True,
            "no_warnings": True,
        }

        frame_paths: List[str] = []
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([reel_url])
            frame_paths = _extract_frames(video_path, frames_dir)
        except Exception as e:
            # Surface a clear error so the API layer can report details back
            raise RuntimeError(f"Video download failed for this URL: {e}") from e

        if not frame_paths:
            # Fall back to URL-only summary if no frames extracted
            resp = client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "You are helping me build a personal knowledge base from short social media reels.\n"
                                    "Please infer the content of this reel/video and summarize it in 2-3 short sentences, "
                                    "focusing on practical information (e.g. destination, food place, tips, prices, etc.).\n\n"
                                    f"Reel URL: {reel_url}"
                                ),
                            }
                        ],
                    }
                ],
            )
            return resp.output[0].content[0].text.strip()

        # Limit number of frames sent to GPT-4o to keep request light
        max_frames = 8
        selected_frames = frame_paths[:max_frames]

        image_contents = []
        for frame_path in selected_frames:
            with open(frame_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            image_contents.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                }
            )

        resp = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are helping me build a personal knowledge base from short social media reels.\n"
                                "Look at these frames extracted from a single reel (1 frame every ~2 seconds) "
                                "and summarize what the reel is about in 2-3 short sentences. "
                                "Focus on destination, venue/restaurant names, activities, tips, and any prices or recommendations."
                            ),
                        },
                        *image_contents,
                    ],
                }
            ],
        )
        return resp.output[0].content[0].text.strip()


def auto_tag_text(client: OpenAI, text: str) -> List[str]:
    """
    Ask the LLM for a small set of tags like travel / restaurant / hotel / street-food / tips, etc.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a tagging assistant. Given a reel transcript or summary, "
                    "return 3-5 short tags (like 'travel', 'restaurant', 'hotel', "
                    "'street-food', 'Bali', 'Goa', 'budget', etc.) as a comma-separated list."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    # de-duplicate and normalize
    unique = list(dict.fromkeys([t.lower() for t in tags]))
    return unique


def embed_text(client: OpenAI, text: str) -> List[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding

