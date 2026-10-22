"""In-process fake of an OpenAI-compatible streaming endpoint, for client and sweep tests."""

import asyncio
import json

import httpx


class _SSE(httpx.AsyncByteStream):
    def __init__(self, events, delay):
        self.events, self.delay = events, delay

    async def __aiter__(self):
        for e in self.events:
            await asyncio.sleep(self.delay)
            yield e


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


class FakeEngine:
    """Per-token delay grows with the number of in-flight requests."""

    def __init__(self, base_delay=0.002, per_inflight=0.0, status=200, honor_ignore_eos=True):
        self.base_delay, self.per_inflight = base_delay, per_inflight
        self.status, self.honor_ignore_eos = status, honor_ignore_eos
        self.inflight = 0
        self.requests: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "fake-model"}]})
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx.Response(self.status, text="boom")
        n = body["max_tokens"]
        if not body.get("ignore_eos") or not self.honor_ignore_eos:
            n = max(1, n - 1)
        self.inflight += 1
        delay = self.base_delay + self.per_inflight * self.inflight
        prompt = sum(len(m["content"].split()) for m in body["messages"])
        events = [_sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]})]
        events += [_sse({"choices": [{"delta": {"content": f"t{i} "}}]}) for i in range(n)]
        events += [
            _sse({"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": n}}),
            b"data: [DONE]\n\n",
        ]

        class Tracked(_SSE):
            async def __aiter__(inner):
                try:
                    async for e in super().__aiter__():
                        yield e
                finally:
                    self.inflight -= 1

        return httpx.Response(200, stream=Tracked(events, delay))
