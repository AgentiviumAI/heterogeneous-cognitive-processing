from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from .providers import ModelTargetSpec, ProviderRegistry, ProviderSpec, ResolvedModelTarget

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            default = match.group(2)
            resolved = os.environ.get(name, default)
            if resolved is None:
                raise RuntimeError(
                    f"Environment variable {name!r} is required by the model config."
                )
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


@dataclass(frozen=True)
class ModelCatalog:
    providers: dict[str, ProviderSpec]
    targets: dict[str, ModelTargetSpec]

    @classmethod
    def load(cls, path: str | Path) -> "ModelCatalog":
        config_path = Path(path)
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        raw = _expand_env(raw)

        providers: dict[str, ProviderSpec] = {}
        for name, item in (raw.get("providers") or {}).items():
            providers[name] = ProviderSpec(
                name=name,
                adapter=str(item.get("adapter", "litellm")),
                litellm_prefix=item.get("litellm_prefix"),
                api_base=item.get("api_base"),
                api_key_env=item.get("api_key_env"),
                api_keys_env=item.get("api_keys_env"),
                api_key_optional=bool(item.get("api_key_optional", False)),
                default_params=dict(item.get("default_params") or {}),
                router_kwargs=dict(item.get("router_kwargs") or {}),
                quota_scope=str(item.get("quota_scope", "shared")),
                shared_tpm=(int(item["shared_tpm"]) if item.get("shared_tpm") is not None else None),
                shared_rpm=(int(item["shared_rpm"]) if item.get("shared_rpm") is not None else None),
                per_key_tpm=(int(item["per_key_tpm"]) if item.get("per_key_tpm") is not None else None),
                per_key_rpm=(int(item["per_key_rpm"]) if item.get("per_key_rpm") is not None else None),
                rate_limit_headroom=float(item.get("rate_limit_headroom", 0.80)),
                adaptive_rate_limit_recovery=bool(item.get("adaptive_rate_limit_recovery", False)),
                rate_limit_recovery_attempts=int(item.get("rate_limit_recovery_attempts", 8)),
                rate_limit_default_wait_seconds=float(item.get("rate_limit_default_wait_seconds", 5.0)),
                rate_limit_max_wait_seconds=float(item.get("rate_limit_max_wait_seconds", 60.0)),
            )

        targets: dict[str, ModelTargetSpec] = {}
        for name, item in (raw.get("targets") or {}).items():
            targets[name] = ModelTargetSpec(
                name=name,
                provider=str(item["provider"]),
                model=str(item["model"]),
                temperature=float(item.get("temperature", 0.0)),
                params=dict(item.get("params") or {}),
            )

        return cls(providers=providers, targets=targets)

    def resolve_target(
        self,
        target_name: str,
        registry: ProviderRegistry | None = None,
    ) -> ResolvedModelTarget:
        registry = registry or ProviderRegistry()
        try:
            target = self.targets[target_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown target {target_name!r}. Available: {', '.join(sorted(self.targets))}"
            ) from exc
        try:
            provider = self.providers[target.provider]
        except KeyError as exc:
            raise ValueError(
                f"Target {target_name!r} references unknown provider {target.provider!r}."
            ) from exc
        return registry.resolve(provider, target)

    def resolve_inline(
        self,
        *,
        provider_name: str,
        model: str,
        temperature: float = 0.0,
        params: dict[str, Any] | None = None,
        registry: ProviderRegistry | None = None,
    ) -> ResolvedModelTarget:
        registry = registry or ProviderRegistry()
        try:
            provider = self.providers[provider_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown provider {provider_name!r}. Available: "
                f"{', '.join(sorted(self.providers))}"
            ) from exc
        target = ModelTargetSpec(
            name=f"{provider_name}:{model}",
            provider=provider_name,
            model=model,
            temperature=temperature,
            params=params or {},
        )
        return registry.resolve(provider, target)
