"""
LLM Dispatcher — single point of entry for all LLM calls in the simulation.

Uses an OpenAI-compatible API (LongCat provider).
All calls enforce structured JSON output validated against a Pydantic schema.

Exposes both sync (``call``) and async (``acall``) entry points.
The simulation uses ``acall`` so that all LLM requests within a phase
run concurrently via ``asyncio.gather``.

Key rotation
~~~~~~~~~~~~
When ``LLM_API_KEYS`` is set (comma-separated), the dispatcher creates
one ``AsyncOpenAI`` client per key and round-robins across them on every
``acall``.  This distributes token usage across multiple quota buckets so
that a large run (e.g. 100×50 or a 20-seed Monte Carlo) doesn't hit a
single key's daily cap.  The sync ``call`` always uses the first key.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import threading
from typing import List, Type, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# Default concurrency cap for async calls.  Override via env or LLMDispatcher(...).
_DEFAULT_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "10"))


def _load_api_keys() -> List[str]:
    """Load API keys from environment.

    Reads ``LLM_API_KEYS`` first (comma-separated list).  Falls back to
    the single ``LLM_API_KEY`` for backward compatibility.
    """
    multi = os.getenv("LLM_API_KEYS", "")
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys
    single = os.getenv("LLM_API_KEY", "")
    return [single] if single else []


class LLMDispatcher:
    """
    Wraps the LLM API.  Every call:
      1. Sends a system + user prompt
      2. Requests JSON output
      3. Parses and validates against a Pydantic schema
      4. Raises on validation failure (never silently swallows bad output)

    Provides:
      * ``call(...)``  — synchronous entry point (ad-hoc use, tests)
      * ``acall(...)`` — async entry point used by the simulation runner

    When multiple API keys are configured (``LLM_API_KEYS``), ``acall``
    round-robins across them to spread quota usage.
    """

    def __init__(self, max_concurrency: int = _DEFAULT_MAX_CONCURRENCY):
        keys = _load_api_keys()
        if not keys:
            raise RuntimeError(
                "No LLM API key configured.  Set LLM_API_KEYS (comma-separated) "
                "or LLM_API_KEY in .env."
            )

        base_url = os.getenv("LLM_BASE_URL")
        self.model = os.getenv("LLM_MODEL_NAME", "LongCat-Flash-Chat")
        self.n_keys = len(keys)

        # Sync client — always uses the first key (sync path is rare).
        self.client = OpenAI(api_key=keys[0], base_url=base_url, timeout=90.0)

        # Async clients — one per key for round-robin.
        self._aclients: List[AsyncOpenAI] = [
            AsyncOpenAI(api_key=k, base_url=base_url, timeout=90.0)
            for k in keys
        ]
        # Thread-safe round-robin counter.
        self._rr = itertools.cycle(range(len(self._aclients)))
        self._rr_lock = threading.Lock()

        # Concurrency gate for async calls.  Lazily created inside the running
        # event loop so we don't bind to a loop at import time.
        self._max_concurrency = max_concurrency
        self._semaphore: asyncio.Semaphore | None = None

        print(f"  [dispatcher] {self.n_keys} API key(s) loaded, "
              f"concurrency cap={max_concurrency}")

    # ── semaphore lifecycle ──────────────────────────────────

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        return self._semaphore

    # ── response parsing ─────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract the first valid JSON object from text.
        Handles thinking models that emit reasoning prose before the JSON block.
        """
        # Strip markdown code fences
        if "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    return block
        # Find outermost { ... }
        start = text.find("{")
        if start == -1:
            return text
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text[start:]  # unclosed — let json.loads report the error

    @staticmethod
    def _parse_response(content: str | None, raw_response, schema: Type[T]) -> T:
        """Shared post-processing for sync + async responses."""
        if content is None:
            content = getattr(raw_response.choices[0].message, "reasoning_content", None)
        if content is None:
            raise ValueError(f"LLM returned empty content. Full response: {raw_response}")

        raw = LLMDispatcher._extract_json(content.strip())

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw output:\n{raw}") from e

        try:
            if hasattr(schema, "model_validate"):
                return schema.model_validate(data)
            return schema.parse_obj(data)
        except ValidationError as e:
            raise ValueError(
                f"LLM output failed schema validation ({schema.__name__}):\n{e}\n"
                f"Raw output:\n{raw}"
            ) from e

    @staticmethod
    def _build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
        enforced = user_prompt + (
            "\n\nIMPORTANT: Your entire response must be valid JSON only. "
            "No markdown, no explanation outside the JSON."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enforced},
        ]

    # ── public API ───────────────────────────────────────────

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        max_tokens: int = 3000,
        temperature: float = 0.7,
    ) -> T:
        """Synchronous call.  Prefer ``acall`` inside the simulation loop."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(system_prompt, user_prompt),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return self._parse_response(content, response, schema)

    def _next_aclient(self) -> AsyncOpenAI:
        """Pick the next async client via thread-safe round-robin."""
        with self._rr_lock:
            idx = next(self._rr)
        return self._aclients[idx]

    async def acall(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        max_tokens: int = 3000,
        temperature: float = 0.7,
    ) -> T:
        """Async call, gated by a shared concurrency semaphore.

        When multiple API keys are configured, each call is dispatched to
        the next key in round-robin order so that quota usage is spread
        evenly across all keys.
        """
        aclient = self._next_aclient()
        async with self._get_semaphore():
            response = await aclient.chat.completions.create(
                model=self.model,
                messages=self._build_messages(system_prompt, user_prompt),
                max_tokens=max_tokens,
                temperature=temperature,
            )
        content = response.choices[0].message.content
        return self._parse_response(content, response, schema)


# Module-level singleton — import and use directly
dispatcher = LLMDispatcher()
