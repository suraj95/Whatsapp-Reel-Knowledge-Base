import os

from celery import Celery
from dotenv import find_dotenv, load_dotenv

# Celery runs as a separate process from FastAPI; load .env explicitly here.
load_dotenv(find_dotenv(usecwd=True))

from backend.observability import configure_observability

configure_observability()

broker_url = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
result_backend = os.getenv("CELERY_RESULT_BACKEND", broker_url)

# include ensures task modules are loaded at worker boot.
celery_app = Celery(
    "travel_reels",
    broker=broker_url,
    backend=result_backend,
    include=["backend.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
)

# Explicitly discover backend task modules as a safety net.
celery_app.autodiscover_tasks(["backend"])

