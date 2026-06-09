"""LMetric multi-instance router for tokenspeed.

Routes requests across multiple ``ts serve`` instances using the LMetric
scoring function: ``score = new_prefill_tokens × batch_size``.

The router is zero-config: it tracks batch_size locally (in-flight requests
per instance) and estimates prefix cache affinity via prefix hashing — no
server-side changes required.

Usage::

    ts router --instance-urls http://host1:8000 http://host2:8000 --port 9000

Architecture::

    Client
      ↓
    LMetric Router  :9000
      ├─ score(instance_0) = new_prefill_est × inflight_0
      ├─ score(instance_1) = new_prefill_est × inflight_1
      └─ route to argmin(score)
          ↓
    ts serve :8000  (instance_0 or instance_1)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("tokenspeed.router")

_STREAM_CHUNK_SIZE = 8192
_PROXY_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=600)
_HEALTH_CHECK_INTERVAL = 5.0


# ---------------------------------------------------------------------------
# Prefix cache affinity tracker
# ---------------------------------------------------------------------------


class PrefixAffinityTracker:
    """Track which instance likely has a given prefix cached.

    Uses a hash of the first ``prefix_window`` tokens to identify shared
    prefixes. Maintains a routing history: if prefix P was last routed to
    instance I, that instance likely has P in its KV cache.
    """

    def __init__(self, prefix_window: int = 256):
        self.prefix_window = prefix_window
        # prefix_hash -> (instance_idx, last_routed_time)
        self._affinity: dict[str, tuple[int, float]] = {}
        self._ttl = 300.0  # 5 min — cache entries evict after this

    def _hash_prefix(self, text: str) -> str:
        prefix = text[: self.prefix_window * 4]  # ~4 chars per token estimate
        return hashlib.md5(prefix.encode(), usedforsecurity=False).hexdigest()[:16]

    def get_affinity(self, prompt_text: str) -> int | None:
        """Return instance index that likely has this prefix cached, or None."""
        h = self._hash_prefix(prompt_text)
        entry = self._affinity.get(h)
        if entry is None:
            return None
        idx, ts = entry
        if time.monotonic() - ts > self._ttl:
            del self._affinity[h]
            return None
        return idx

    def record(self, prompt_text: str, instance_idx: int) -> None:
        h = self._hash_prefix(prompt_text)
        self._affinity[h] = (instance_idx, time.monotonic())

    def estimate_new_prefill(
        self, prompt_text: str, instance_idx: int, total_tokens_est: int
    ) -> int:
        """Estimate new prefill tokens if routed to instance_idx.

        If this prefix was recently routed to instance_idx, assume full cache
        hit (new_prefill ≈ suffix only). Otherwise assume no cache hit
        (new_prefill ≈ total_tokens).
        """
        affinity = self.get_affinity(prompt_text)
        if affinity == instance_idx:
            # Estimate: cached prefix covers ~prefix_window tokens worth,
            # only the suffix needs prefill
            return max(total_tokens_est - self.prefix_window, 1)
        return total_tokens_est


# ---------------------------------------------------------------------------
# Instance state
# ---------------------------------------------------------------------------


@dataclass
class InstanceState:
    url: str
    inflight: int = 0
    healthy: bool = True
    last_health_check: float = 0.0


# ---------------------------------------------------------------------------
# LMetric Router
# ---------------------------------------------------------------------------


class LMetricRouter:
    """Routes requests to the instance with the lowest LMetric score.

    score(instance_i) = new_prefill_tokens_i × (inflight_i + 1)

    - new_prefill_tokens: estimated from prefix affinity tracking
    - inflight: locally tracked in-flight request count (no polling needed)
    """

    def __init__(
        self,
        instance_urls: list[str],
        prefix_window: int = 256,
    ):
        self.instances = [InstanceState(url=u) for u in instance_urls]
        self.affinity = PrefixAffinityTracker(prefix_window=prefix_window)
        self._session: aiohttp.ClientSession | None = None
        self._round_robin_idx = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_PROXY_TIMEOUT)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _extract_prompt(self, body: dict) -> str:
        """Extract prompt text from request body for prefix hashing."""
        # /v1/chat/completions
        messages = body.get("messages")
        if messages:
            parts = []
            for m in messages:
                c = m.get("content", "")
                if isinstance(c, str):
                    parts.append(c)
            return " ".join(parts)
        # /v1/completions
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            return " ".join(str(p) for p in prompt)
        return str(prompt)

    def _estimate_tokens(self, text: str) -> int:
        return max(len(text) // 4, 1)

    def select_instance(self, body: dict) -> int:
        """Select the instance with the lowest LMetric score."""
        prompt = self._extract_prompt(body)
        total_tokens = self._estimate_tokens(prompt)

        best_idx = 0
        best_score = float("inf")

        for i, inst in enumerate(self.instances):
            if not inst.healthy:
                continue
            new_prefill = self.affinity.estimate_new_prefill(
                prompt, i, total_tokens
            )
            score = new_prefill * (inst.inflight + 1)
            if score < best_score:
                best_score = score
                best_idx = i

        self.affinity.record(prompt, best_idx)
        return best_idx

    def select_instance_round_robin(self) -> int:
        """Fallback: simple round-robin."""
        healthy = [i for i, inst in enumerate(self.instances) if inst.healthy]
        if not healthy:
            return 0
        idx = healthy[self._round_robin_idx % len(healthy)]
        self._round_robin_idx += 1
        return idx

    async def check_health(self):
        """Periodic health check for all instances."""
        session = await self._get_session()
        for inst in self.instances:
            try:
                async with session.get(
                    f"{inst.url}/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    inst.healthy = resp.status == 200
            except Exception:
                inst.healthy = False
            inst.last_health_check = time.monotonic()

    def get_stats(self) -> dict:
        return {
            "instances": [
                {
                    "url": inst.url,
                    "inflight": inst.inflight,
                    "healthy": inst.healthy,
                }
                for inst in self.instances
            ],
            "affinity_entries": len(self.affinity._affinity),
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app(router: LMetricRouter, mode: str = "lmetric") -> FastAPI:
    app = FastAPI(title="TokenSpeed LMetric Router")

    @app.on_event("startup")
    async def startup():
        await router.check_health()

        async def _health_loop():
            while True:
                await asyncio.sleep(_HEALTH_CHECK_INTERVAL)
                await router.check_health()

        asyncio.create_task(_health_loop())

    @app.on_event("shutdown")
    async def shutdown():
        await router.close()

    # --- Routing endpoints ---

    async def _route_request(request: Request) -> StreamingResponse | Response:
        body_bytes = await request.body()
        try:
            body = await request.json()
        except Exception:
            body = {}

        if mode == "lmetric":
            idx = router.select_instance(body)
        else:
            idx = router.select_instance_round_robin()

        inst = router.instances[idx]
        inst.inflight += 1

        target_url = f"{inst.url.rstrip('/')}{request.url.path}"
        if request.url.query:
            target_url = f"{target_url}?{request.url.query}"
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }

        session = await router._get_session()
        try:
            resp = await session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body_bytes,
                timeout=_PROXY_TIMEOUT,
            )
        except Exception as e:
            inst.inflight = max(0, inst.inflight - 1)
            return JSONResponse(
                {"error": f"upstream {inst.url} unreachable: {e}"},
                status_code=502,
            )

        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:

            async def _stream():
                try:
                    async for chunk in resp.content.iter_chunked(_STREAM_CHUNK_SIZE):
                        yield chunk
                finally:
                    resp.release()
                    inst.inflight = max(0, inst.inflight - 1)

            return StreamingResponse(
                _stream(),
                status_code=resp.status,
                media_type="text/event-stream",
            )

        try:
            data = await resp.read()
            return Response(
                content=data,
                status_code=resp.status,
                media_type=content_type or "application/json",
            )
        finally:
            inst.inflight = max(0, inst.inflight - 1)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _route_request(request)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _route_request(request)

    @app.api_route("/generate", methods=["GET", "POST"])
    async def generate(request: Request):
        return await _route_request(request)

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await _route_request(request)

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await _route_request(request)

    # --- Health / stats ---

    @app.get("/health")
    async def health():
        healthy = any(inst.healthy for inst in router.instances)
        return Response(
            content="OK" if healthy else "No healthy instances",
            status_code=200 if healthy else 503,
        )

    @app.get("/v1/models")
    async def models(request: Request):
        for inst in router.instances:
            if inst.healthy:
                session = await router._get_session()
                try:
                    async with session.get(f"{inst.url}/v1/models") as resp:
                        data = await resp.read()
                        return Response(
                            content=data,
                            media_type="application/json",
                        )
                except Exception:
                    continue
        return JSONResponse({"error": "no healthy instance"}, status_code=503)

    @app.get("/router/stats")
    async def stats():
        return JSONResponse(router.get_stats())

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    instance_urls: list[str],
    host: str = "0.0.0.0",
    port: int = 9000,
    mode: str = "lmetric",
    prefix_window: int = 256,
):
    router = LMetricRouter(
        instance_urls=instance_urls,
        prefix_window=prefix_window,
    )
    app = create_app(router, mode=mode)
    logger.info(
        "LMetric router starting on %s:%d, mode=%s, instances=%s",
        host,
        port,
        mode,
        instance_urls,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
