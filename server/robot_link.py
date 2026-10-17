"""Server -> Agent link. Re-transmits a command with the SAME id on network errors,
so the Agent's id memory guarantees it runs at most once."""
import asyncio
import ipaddress
import time
import uuid

import httpx

from common import config, envelope


class AgentLink:
    def __init__(self):
        self.cfg = config.load()
        a = self.cfg["agent"]
        self.port = a["port"]
        self.http = httpx.Client(timeout=a["request_timeout_s"])
        self.host = None
        self.health = None

    @property
    def base(self):
        return f"http://{self.host}:{self.port}"

    # ---------------------------------------------------------------- connection
    async def discover(self):
        """Config host first, then every address in the configured subnet."""
        d = self.cfg["discovery"]
        hosts = [self.cfg["robot"]["host"]] + [
            str(h) for h in ipaddress.ip_network(d["subnet"]).hosts() if str(h) != self.cfg["robot"]["host"]]
        found = []
        async with httpx.AsyncClient(timeout=d["timeout_s"]) as client:
            async def probe(h):
                try:
                    r = await client.get(f"http://{h}:{self.port}/health")
                    if r.status_code == 200 and r.json().get("service") == "g1ui-agent":
                        found.append({"host": h, **r.json()})
                except (httpx.HTTPError, ValueError):
                    pass
            await asyncio.gather(*(probe(h) for h in hosts))
        return sorted(found, key=lambda f: f["host"] != self.cfg["robot"]["host"])

    def connect(self, host):
        r = self.http.get(f"http://{host}:{self.port}/health")
        r.raise_for_status()
        self.host, self.health = host, r.json()
        return self.health

    def disconnect(self):
        self.host, self.health = None, None

    # ---------------------------------------------------------------- commands
    def send(self, action, params=None, wait=False, wait_timeout_s=60.0):
        if not self.host:
            return envelope.failed("-", "not connected to a robot", status="rejected").model_dump()
        a = self.cfg["agent"]
        body = {"id": uuid.uuid4().hex, "action": action, "params": params or {}}
        resp, err = None, None
        for _ in range(a["retries"]):
            try:
                resp = self.http.post(f"{self.base}/cmd", json=body).json()
                break
            except httpx.TransportError as e:
                err = e
                time.sleep(a["retry_gap_s"])
        if resp is None:
            return envelope.failed(body["id"], f"agent unreachable after {a['retries']} tries: {err}").model_dump()
        if wait and resp.get("status") == "running":
            resp = self.wait(body["id"], wait_timeout_s)
        return resp

    def wait(self, cmd_id, timeout_s):
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            try:
                r = self.http.get(f"{self.base}/cmd/{cmd_id}").json()
                if r.get("status") != "running":
                    return r
            except httpx.TransportError:
                pass
            time.sleep(0.2)
        return envelope.failed(cmd_id, f"still running after {timeout_s}s", status="running").model_dump()

    def state(self):
        return self.http.get(f"{self.base}/state").json()
