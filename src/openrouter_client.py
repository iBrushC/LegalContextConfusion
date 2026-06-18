"""openrouter_client.py — minimal OpenRouter chat-completions client.

Stdlib-only HTTP client used by run_models.py. One entry point,
call_openrouter(), which always returns a uniform dict:

    {"content": str|None, "usage": dict, "latency_s": float, "error": str|None}

Transient HTTP/network failures are retried with exponential backoff; on final
failure the dict carries content=None and a human-readable error string instead
of raising, so the caller can record it as a failed run row.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# HTTP status codes worth retrying (timeouts, conflicts, rate limits, 5xx).
_RETRY_CODES = (408, 409, 429, 500, 502, 503, 504)


def call_openrouter(slug: str, messages: list[dict], api_key: str,
                    max_tokens: int, temperature: float, timeout: float,
                    retries: int, reasoning: dict | None = None) -> dict:
    """POST a chat completion to OpenRouter and return a uniform result dict."""
    payload = {
        "model": slug,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},  # ask OpenRouter to return cost when available
    }
    if reasoning is not None:
        # control reasoning-model thinking budget so it doesn't eat max_tokens
        payload["reasoning"] = reasoning
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/iBrushC/LegalContextConfusion",
        "X-Title": "LegalContextConfusion",
    }
    for attempt in range(retries + 1):
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(API_URL, data=body, headers=headers,
                                         method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return {
                "content": payload["choices"][0]["message"]["content"],
                "usage": payload.get("usage", {}) or {},
                "latency_s": round(time.monotonic() - t0, 3),
                "error": None,
            }
        except urllib.error.HTTPError as e:
            try:
                detail = e.read()[:300].decode("utf-8", "ignore")
            except Exception:
                detail = ""
            err = f"HTTP {e.code}: {detail}"
            if e.code in _RETRY_CODES and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return {"content": None, "usage": {},
                    "latency_s": round(time.monotonic() - t0, 3), "error": err}
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException, json.JSONDecodeError) as e:
            # transient network / truncated-read / malformed-response-body errors
            # (incl. IncompleteRead and a half-streamed JSON envelope) — retry
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return {"content": None, "usage": {},
                    "latency_s": round(time.monotonic() - t0, 3),
                    "error": f"{type(e).__name__}: {e}"}
        except Exception as e:
            # anything unexpected (e.g. a malformed response shape) — record it,
            # never let one bad response abort the whole sweep
            return {"content": None, "usage": {},
                    "latency_s": round(time.monotonic() - t0, 3),
                    "error": f"{type(e).__name__}: {e}"}
    return {"content": None, "usage": {}, "latency_s": 0.0, "error": "no attempts made"}
