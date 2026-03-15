from typing import List

from openai import OpenAI


def fake_transcript_from_reel(url: str) -> str:
    """
    Stub for fetching transcript from a reel URL.
    Replace with real logic (e.g., using WhatsApp/Instagram APIs,
    or a downloader + Whisper transcription).
    """
    return (
        f"Transcript of reel at {url}. It talks about great street food in Goa with "
        f"local restaurants and some travel tips."
    )


def summarize_text(client: OpenAI, text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You summarize social media reels in 2-3 short sentences.",
            },
            {"role": "user", "content": f"Summarize this reel transcript:\n\n{text}"},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


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

