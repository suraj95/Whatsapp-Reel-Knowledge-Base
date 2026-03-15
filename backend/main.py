import os
import uuid

from fastapi import FastAPI
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone

from .helpers import (
    auto_tag_text,
    embed_text,
    fake_transcript_from_reel,
    summarize_text,
)
from .models import (
    AddReelRequest,
    AddReelResponse,
    QueryRequest,
    QueryResponse,
    ReelResult,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment (e.g. in a .env file).")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Set PINECONE_API_KEY in your environment (e.g. in a .env file).")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Vector DB (Pinecone) ----
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "whatsapp-reels"

existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    raise RuntimeError(
        f"Pinecone index '{INDEX_NAME}' not found. "
        "Create it in the Pinecone console with dimension=1536 and metric='cosine'."
    )

index = pc.Index(INDEX_NAME)

# ---- FastAPI app ----
app = FastAPI(title="WhatsApp Reel Knowledge Base")


@app.post("/reels", response_model=AddReelResponse)
def add_reel(payload: AddReelRequest):
    # 1. Get transcript (or use Whisper if you have audio file)
    transcript = fake_transcript_from_reel(payload.url)

    # 2. Summarize with LLM
    summary = summarize_text(client, transcript)

    # 3. Auto tag
    auto_tags = auto_tag_text(client, summary + "\n\n" + transcript)
    if payload.manual_tags:
        auto_tags = list(
            dict.fromkeys(auto_tags + [t.lower() for t in payload.manual_tags])
        )

    # 4. Build embedding using transcript + summary + tags
    doc_text = (
        f"URL: {payload.url}\n"
        f"SUMMARY: {summary}\n"
        f"TAGS: {', '.join(auto_tags)}\n"
        f"TRANSCRIPT: {transcript}"
    )
    embedding = embed_text(client, doc_text)

    reel_id = str(uuid.uuid4())

    # 5. Store in Pinecone
    index.upsert(
        vectors=[
            {
                "id": reel_id,
                "values": embedding,
                "metadata": {
                    "url": payload.url,
                    "summary": summary,
                    "tags": auto_tags,
                    "doc_text": doc_text,
                },
            }
        ]
    )

    return AddReelResponse(
        reel_id=reel_id,
        summary=summary,
        auto_tags=auto_tags,
    )


@app.post("/query", response_model=QueryResponse)
def query_reels(payload: QueryRequest):
    """
    Example queries:
    - 'show restaurants we saved in Goa'
    - 'any reels about Bali?'
    - 'cheap street food ideas?'
    """
    # 1. Embed query
    query_embedding = embed_text(client, payload.query)

    # 2. Vector search in Pinecone
    res = index.query(
        vector=query_embedding,
        top_k=payload.top_k,
        include_metadata=True,
    )

    if not res.matches:
        return QueryResponse(results=[])

    results: List[ReelResult] = []
    for match in res.matches:
        meta = match.metadata or {}
        score = float(match.score) if match.score is not None else 0.0
        results.append(
            ReelResult(
                reel_id=match.id,
                url=meta.get("url", ""),
                summary=meta.get("summary", ""),
                auto_tags=meta.get("tags", []),
                score=score,
            )
        )

    return QueryResponse(results=results)

