"""压测脚本:  locust -f locustfile.py --host http://127.0.0.1:8000"""
from locust import HttpUser, between, task


class DanmakuUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task
    def predict(self):
        self.client.post("/predict", json={"text": "前方高能预警,这波操作太秀了"})
