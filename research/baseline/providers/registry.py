from __future__ import annotations

from typing import Iterable

from .base import ModelTargetSpec, ProviderAdapter, ProviderSpec, ResolvedModelTarget
from .litellm import LiteLLMNativeAdapter, OpenAICompatibleAdapter


class ProviderRegistry:
    def __init__(self, adapters: Iterable[ProviderAdapter] | None = None) -> None:
        builtins = adapters or (LiteLLMNativeAdapter(), OpenAICompatibleAdapter())
        self._adapters = {adapter.name: adapter for adapter in builtins}

    def register(self, adapter: ProviderAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def resolve(
        self,
        provider: ProviderSpec,
        target: ModelTargetSpec,
    ) -> ResolvedModelTarget:
        try:
            adapter = self._adapters[provider.adapter]
        except KeyError as exc:
            known = ", ".join(sorted(self._adapters))
            raise ValueError(
                f"Unknown provider adapter {provider.adapter!r}; available: {known}"
            ) from exc
        return adapter.resolve(provider, target)
