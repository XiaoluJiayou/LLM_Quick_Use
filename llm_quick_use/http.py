"""Bounded client for OpenAI-compatible chat endpoints; no SDK required."""

import json
import math
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatClient:
    endpoint: str
    model: str
    api_key: Optional[str] = field(default=None, repr=False)
    timeout: float = 60.0
    max_attempts: int = 3
    temperature: float = 0.7
    max_tokens: int = 256

    def __post_init__(self):
        url = urllib.parse.urlsplit(self.endpoint)
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("endpoint must be an HTTP(S) URL without credentials, query or fragment")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if (not isinstance(self.max_attempts, int) or not isinstance(self.max_tokens, int)
                or self.max_attempts < 1 or self.max_tokens < 1):
            raise ValueError("max_attempts and max_tokens must be positive")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    def public_config(self):
        """Configuration safe to persist: authentication never enters journals."""
        return {"endpoint": self.endpoint, "model": self.model,
                "temperature": self.temperature, "max_tokens": self.max_tokens}

    def complete(self, messages):
        payload = json.dumps({"model": self.model, "messages": messages,
                              "temperature": self.temperature,
                              "max_tokens": self.max_tokens}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            key = self.api_key.strip()
            headers["Authorization"] = key if key.lower().startswith("bearer ") else "Bearer " + key
        error = "request failed"
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(self.endpoint, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    status = response.getcode()
                    if status is not None and not 200 <= status < 300:
                        raise urllib.error.HTTPError(self.endpoint, status, "HTTP error", {}, None)
                    raw = response.read()
                try:
                    result = json.loads(raw)
                    if not isinstance(result, dict):
                        raise ValueError("chat response must be an object")
                    choices = result.get("choices")
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        raise ValueError("choices must contain a completion object")
                    choice = choices[0]
                    if choice.get("finish_reason") in {"length", "content_filter"}:
                        return {"status": "error", "content": "", "error": "Chat completion was truncated or filtered", "attempts": attempt}
                    message = choice.get("message")
                    if not isinstance(message, dict):
                        raise ValueError("message must be an object")
                    content = message.get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("empty content")
                except (ValueError, TypeError, KeyError, IndexError, UnicodeDecodeError):
                    return {"status": "error", "content": "", "error": "Invalid chat JSON response", "attempts": attempt}
                return {"status": "ok", "content": content, "error": "", "attempts": attempt}
            except urllib.error.HTTPError as exc:
                error = "HTTP {}".format(exc.code)
                retry = exc.code == 429 or 500 <= exc.code <= 599
                if exc.fp is not None:
                    exc.close()
                if not retry:
                    break
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
                error = "Network error ({})".format(type(exc).__name__)
            if attempt < self.max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
        return {"status": "error", "content": "", "error": error, "attempts": attempt}
