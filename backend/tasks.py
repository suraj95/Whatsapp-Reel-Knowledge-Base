from .celery_app import celery_app
from .services.ingestion_worker import process_reel_ingestion


@celery_app.task(name="backend.tasks.process_reel_ingestion_task")
def process_reel_ingestion_task(job_id: str, reel_url: str, manual_tags=None):
    return process_reel_ingestion(job_id=job_id, reel_url=reel_url, manual_tags=manual_tags)

