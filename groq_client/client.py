"""
groq_client/client.py
Manages multiple Groq API keys with round-robin rotation, retry logic,
and token usage tracking.
"""

import time
import random
from dataclasses import dataclass, field
from groq import Groq


@dataclass
class UsageStats:
    """Tracks token usage per API key."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class GroqClientManager:
    """
    Manages multiple Groq API keys with round-robin rotation.
    Automatically retries with next key on rate-limit or transient errors.
    Permanently removes keys that fail authentication.
    """

    def __init__(self, api_keys: list[str], model: str = "llama-3.1-8b-instant"):
        if not api_keys:
            raise ValueError("At least one Groq API key is required.")
        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.clients = [Groq(api_key=key) for key in self.api_keys]
        self.model = model
        self.current_index = 0
        # Usage tracking per key
        self.usage: dict[str, UsageStats] = {k: UsageStats() for k in self.api_keys}

    def _next_client(self) -> tuple[Groq, str]:
        """Return next (client, key) in round-robin order."""
        client = self.clients[self.current_index]
        key = self.api_keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.clients)
        return client, key

    def _remove_key(self, key: str) -> None:
        """Permanently remove a key that failed auth."""
        try:
            idx = self.api_keys.index(key)
            self.clients.pop(idx)
            self.api_keys.pop(idx)
            if self.clients:
                self.current_index = self.current_index % len(self.clients)
            else:
                self.current_index = 0
        except ValueError:
            pass  # already removed

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> str:
        """
        Call Groq API with automatic key rotation and retry.
        Lower temperature (0.2) for analytical tasks by default.
        Returns the text content of the response.
        """
        if not self.clients:
            raise RuntimeError("No valid Groq API keys available.")

        last_error = None
        total_tries = max_retries * max(len(self.clients), 1)

        for attempt in range(total_tries):
            if not self.clients:
                raise RuntimeError("All Groq API keys exhausted (auth errors).")

            client, current_key = self._next_client()
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # Track usage
                if hasattr(response, "usage") and response.usage and current_key in self.usage:
                    self.usage[current_key].prompt_tokens += response.usage.prompt_tokens or 0
                    self.usage[current_key].completion_tokens += response.usage.completion_tokens or 0
                    self.usage[current_key].total_calls += 1

                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                if "rate_limit" in err_str or "429" in err_str:
                    wait = 2 ** (attempt % 4) + random.uniform(0, 1)
                    time.sleep(wait)
                    continue

                if "auth" in err_str or "401" in err_str or "invalid api key" in err_str:
                    self._remove_key(current_key)
                    continue

                if "404" in err_str or "400" in err_str:
                    raise RuntimeError(f"Groq API error (model '{self.model}'): {e}") from e

                time.sleep(1)

        raise RuntimeError(f"All Groq API attempts exhausted. Last error: {last_error}")

    def get_usage_summary(self) -> dict:
        """Return token usage summary across all keys."""
        total_prompt = sum(u.prompt_tokens for u in self.usage.values())
        total_completion = sum(u.completion_tokens for u in self.usage.values())
        total_calls = sum(u.total_calls for u in self.usage.values())
        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_calls": total_calls,
            "per_key": {
                f"...{k[-4:]}": {
                    "prompt": u.prompt_tokens,
                    "completion": u.completion_tokens,
                    "calls": u.total_calls,
                }
                for k, u in self.usage.items()
            },
        }
