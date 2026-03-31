import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pinecone import Pinecone

from ..helpers import (
    auto_tag_text,
    embed_text,
    enrich_reel_summary,
    extract_reel_metadata_with_yt_dlp,
    format_enrichment_for_metadata,
    summarize_video_with_gpt4o,
)
from ..job_store import set_job_status
from ..models import AddReelResponse, EnrichmentData, IngestionStatus

logger = logging.getLogger(__name__)
INDEX_NAME = "whatsapp-reels"


def _build_clients():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not openai_api_key or not pinecone_api_key:
        raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY are required")

    client = OpenAI(api_key=openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)
    return client, index


def _status_payload(job_id: str, status: IngestionStatus, **kwargs: Any) -> Dict[str, Any]:
    payload = {"job_id": job_id, "status": status.value}
    payload.update(kwargs)
    return payload


async def process_reel_ingestion_async(
    job_id: str,
    reel_url: str,
    manual_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    client, index = _build_clients()
    try:
        set_job_status(job_id, _status_payload(job_id, IngestionStatus.running, stage="summarizing"))
        summary = summarize_video_with_gpt4o(client, reel_url)

        set_job_status(job_id, _status_payload(job_id, IngestionStatus.running, stage="tagging"))
        auto_tags = auto_tag_text(client, summary)
        if manual_tags:
            auto_tags = list(dict.fromkeys(auto_tags + [t.lower() for t in manual_tags]))

        set_job_status(job_id, _status_payload(job_id, IngestionStatus.running, stage="extracting_metadata"))
        reel_metadata = extract_reel_metadata_with_yt_dlp(reel_url)

        set_job_status(job_id, _status_payload(job_id, IngestionStatus.running, stage="enriching"))
        enrichment = await enrich_reel_summary(
            summary,
            reel_url=reel_url,
            reel_metadata=reel_metadata,
        )
        enrichment_metadata, enrichment_json = format_enrichment_for_metadata(enrichment)

        set_job_status(job_id, _status_payload(job_id, IngestionStatus.running, stage="embedding"))
        doc_text = f"URL: {reel_url}\nSUMMARY: {summary}\nTAGS: {', '.join(auto_tags)}\n"
        embedding = embed_text(client, doc_text)
        reel_id = str(uuid.uuid4())

        set_job_status(job_id, _status_payload(job_id, IngestionStatus.running, stage="upserting"))
        index.upsert(
            vectors=[
                {
                    "id": reel_id,
                    "values": embedding,
                    "metadata": {
                        "url": reel_url,
                        "summary": summary,
                        "tags": auto_tags,
                        "doc_text": doc_text,
                        **enrichment_metadata,
                        "enrichment_json": enrichment_json,
                    },
                }
            ]
        )
        result = AddReelResponse(
            reel_id=reel_id,
            summary=summary,
            auto_tags=auto_tags,
            enrichment=EnrichmentData(**enrichment),
        )
        result_payload = result.model_dump()
        set_job_status(
            job_id,
            _status_payload(
                job_id,
                IngestionStatus.completed,
                stage="completed",
                reel_id=reel_id,
                result=result_payload,
            ),
        )
        logger.info("ingestion_completed job_id=%s reel_id=%s", job_id, reel_id)
        return result_payload
    except Exception as ex:
        message = str(ex)[:500]
        set_job_status(
            job_id,
            _status_payload(
                job_id,
                IngestionStatus.failed,
                stage="failed",
                error=message,
            ),
        )
        logger.exception("ingestion_failed job_id=%s error=%s", job_id, ex)
        raise


def process_reel_ingestion(job_id: str, reel_url: str, manual_tags: Optional[List[str]] = None) -> Dict[str, Any]:
    return asyncio.run(process_reel_ingestion_async(job_id=job_id, reel_url=reel_url, manual_tags=manual_tags))

