"""Client for the server-side model services (vision, planner, llm). All share one envelope on POST /infer."""
import uuid

import httpx

from common import config


class ServiceError(RuntimeError):
    pass


class Service:
    def __init__(self, name):
        s = config.load()["services"][name]
        self.name = name
        self.url = f"http://{s['host']}:{s['port']}"
        self.http = httpx.Client(timeout=s["timeout_s"])

    def call(self, action, params):
        """Returns the result dict, or raises ServiceError with the service's message."""
        try:
            r = self.http.post(f"{self.url}/infer", json={"id": uuid.uuid4().hex, "action": action, "params": params})
            body = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise ServiceError(f"{self.name} service unreachable: {e}")
        if not body.get("ok"):
            raise ServiceError(f"{self.name}: {body.get('error')}")
        return body["result"]

    def health(self):
        try:
            return self.http.get(f"{self.url}/health", timeout=2).json()
        except (httpx.HTTPError, ValueError):
            return None
