import unittest
from unittest.mock import patch

from backend.job_store import get_job_status, set_job_status
from backend.models import IngestionStatus


class IngestionFlowTests(unittest.TestCase):
    def test_status_transitions_persist(self):
        with patch("backend.job_store._get_redis_client", return_value=None):
            job_id = "job-test-1"
            set_job_status(job_id, {"job_id": job_id, "status": IngestionStatus.queued.value, "stage": "queued"})
            queued = get_job_status(job_id)
            self.assertEqual(queued["status"], "queued")

            set_job_status(job_id, {"job_id": job_id, "status": IngestionStatus.running.value, "stage": "enriching"})
            running = get_job_status(job_id)
            self.assertEqual(running["status"], "running")
            self.assertEqual(running["stage"], "enriching")

            set_job_status(
                job_id,
                {
                    "job_id": job_id,
                    "status": IngestionStatus.completed.value,
                    "stage": "completed",
                    "reel_id": "reel-1",
                },
            )
            done = get_job_status(job_id)
            self.assertEqual(done["status"], "completed")
            self.assertEqual(done["reel_id"], "reel-1")


if __name__ == "__main__":
    unittest.main()

