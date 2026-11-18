from __future__ import annotations

import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


_ENV_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")
_VALID_QUOTA_SCOPES = {"shared", "per_key"}


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    adapter: str
    litellm_prefix: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    # Optional comma-separated key pool. Each distinct value is exposed to
    # LiteLLM through a process-local generated environment variable so raw
    # secrets never appear in run metadata.
    api_keys_env: str | None = None
    api_key_optional: bool = False
    default_params: dict[str, Any] = field(default_factory=dict)
    router_kwargs: dict[str, Any] = field(default_factory=dict)

    # Provider quota model:
    # - shared: all keys share one provider/org quota.
    # - per_key: every key has its own independent quota (for example when the
    #   keys are known to belong to distinct Groq organizations).
    quota_scope: str = "shared"
    shared_tpm: int | None = None
    shared_rpm: int | None = None
    per_key_tpm: int | None = None
    per_key_rpm: int | None = None
    rate_limit_headroom: float = 0.80

    # LiteLLM already retries/rotates deployments. This extra recovery loop is
    # only entered after LiteLLM has exhausted the pool and raised a final 429.
    adaptive_rate_limit_recovery: bool = False
    rate_limit_recovery_attempts: int = 8
    rate_limit_default_wait_seconds: float = 5.0
    rate_limit_max_wait_seconds: float = 60.0


@dataclass(frozen=True)
class ModelTargetSpec:
    name: str
    provider: str
    model: str
    temperature: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedModelTarget:
    target_name: str
    provider_name: str
    adapter_name: str
    requested_model: str
    litellm_model: str
    temperature: float
    litellm_params: dict[str, Any]
    router_kwargs: dict[str, Any]
    api_key_env: str | None = None
    api_keys_env: str | None = None
    api_base: str | None = None
    deployments: tuple[dict[str, Any], ...] = ()
    key_count: int = 0

    quota_scope: str = "shared"
    configured_tpm: int | None = None
    configured_rpm: int | None = None
    effective_tpm: int | None = None
    effective_rpm: int | None = None
    per_deployment_tpm: int | None = None
    per_deployment_rpm: int | None = None
    rate_limit_headroom: float = 0.80

    adaptive_rate_limit_recovery: bool = False
    rate_limit_recovery_attempts: int = 8
    rate_limit_default_wait_seconds: float = 5.0
    rate_limit_max_wait_seconds: float = 60.0

    def router_config(self, *, retries: int | None = None) -> dict[str, Any]:
        """Build the legacy config consumed by ``lmi.LiteLLMModel``.

        Besides LiteLLM's model_list/router kwargs, this deliberately includes
        LMI's top-level ``rate_limit`` and ``request_limit`` entries. LMI wraps
        every acompletion with a process-global weighted limiter, so answer,
        summary, and agent model instances share one token/request budget and
        wait instead of issuing a burst of requests that the provider rejects.
        """
        router_kwargs = dict(self.router_kwargs)
        if retries is not None:
            router_kwargs["num_retries"] = retries

        model_list = (
            [
                {
                    "model_name": item["model_name"],
                    "litellm_params": dict(item["litellm_params"]),
                }
                for item in self.deployments
            ]
            if self.deployments
            else [
                {
                    "model_name": self.litellm_model,
                    "litellm_params": dict(self.litellm_params),
                }
            ]
        )
        config: dict[str, Any] = {
            "model_list": model_list,
            "router_kwargs": router_kwargs,
        }

        # fhlmi's GLOBAL_LIMITER accepts limits-style strings and accounts for
        # estimated prompt tokens before dispatch plus completion tokens after
        # the response. This is the strict waiting layer that LiteLLM's Router
        # pre-call checks do not provide on their own.
        if self.effective_tpm is not None:
            config["rate_limit"] = {
                self.litellm_model: f"{self.effective_tpm}/minute"
            }
        if self.effective_rpm is not None:
            config["request_limit"] = {
                self.litellm_model: f"{self.effective_rpm}/minute"
            }
        return config

    def public_metadata(self) -> dict[str, Any]:
        # Never materialize or serialize the secret itself.
        return {
            "target": self.target_name,
            "provider": self.provider_name,
            "adapter": self.adapter_name,
            "requested_model": self.requested_model,
            "litellm_model": self.litellm_model,
            "temperature": self.temperature,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "api_keys_env": self.api_keys_env,
            "key_count": self.key_count,
            "params": {
                key: value
                for key, value in self.litellm_params.items()
                if key != "api_key"
            },
            "rate_limit": {
                "quota_scope": self.quota_scope,
                "configured_tpm": self.configured_tpm,
                "configured_rpm": self.configured_rpm,
                "headroom": self.rate_limit_headroom,
                "effective_tpm": self.effective_tpm,
                "effective_rpm": self.effective_rpm,
                "per_deployment_tpm": self.per_deployment_tpm,
                "per_deployment_rpm": self.per_deployment_rpm,
                "lmi_global_limiter": True,
                "adaptive_429_recovery": self.adaptive_rate_limit_recovery,
                "recovery_attempts": self.rate_limit_recovery_attempts,
                "default_wait_seconds": self.rate_limit_default_wait_seconds,
                "max_wait_seconds": self.rate_limit_max_wait_seconds,
            },
            "router_kwargs": dict(self.router_kwargs),
        }


class ProviderAdapter(ABC):
    name: str

    def preflight(self, provider: ProviderSpec) -> None:
        has_single = bool(provider.api_key_env and os.environ.get(provider.api_key_env))
        has_pool = bool(provider.api_keys_env and self._parse_key_pool(provider.api_keys_env))
        if (
            (provider.api_key_env or provider.api_keys_env)
            and not provider.api_key_optional
            and not (has_single or has_pool)
        ):
            accepted = ", ".join(
                name for name in (provider.api_key_env, provider.api_keys_env) if name
            )
            raise RuntimeError(
                f"Provider {provider.name!r} requires at least one API key via: {accepted}."
            )
        if provider.quota_scope not in _VALID_QUOTA_SCOPES:
            raise ValueError(
                f"quota_scope for provider {provider.name!r} must be one of "
                f"{sorted(_VALID_QUOTA_SCOPES)}, got {provider.quota_scope!r}."
            )

    @staticmethod
    def _parse_key_pool(env_name: str | None) -> list[str]:
        if not env_name:
            return []
        raw = os.environ.get(env_name, "")
        if not raw.strip():
            return []
        keys: list[str] = []
        seen: set[str] = set()
        for item in raw.split(","):
            key = item.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        return keys

    @classmethod
    def _api_key_references(cls, provider: ProviderSpec) -> tuple[str, ...]:
        """Return secret-safe LiteLLM env references for all configured keys."""
        pool = cls._parse_key_pool(provider.api_keys_env)
        single = os.environ.get(provider.api_key_env, "") if provider.api_key_env else ""

        values: list[tuple[str | None, str]] = []
        seen: set[str] = set()
        if single.strip():
            seen.add(single.strip())
            values.append((provider.api_key_env, single.strip()))
        for value in pool:
            if value in seen:
                continue
            seen.add(value)
            values.append((None, value))

        if not values:
            return ()

        refs: list[str] = []
        generated_prefix = _ENV_SAFE_RE.sub("_", provider.name.upper()).strip("_") or "PROVIDER"
        pool_index = 0
        for source_env, value in values:
            if source_env is not None:
                refs.append(f"os.environ/{source_env}")
                continue
            pool_index += 1
            generated_env = f"PAPERQA_{generated_prefix}_POOL_KEY_{pool_index}"
            os.environ[generated_env] = value
            refs.append(f"os.environ/{generated_env}")
        return tuple(refs)

    @staticmethod
    def _rate_limits(
        provider: ProviderSpec,
        deployment_count: int,
    ) -> tuple[int | None, int | None, int | None, int | None, int | None, int | None]:
        """Resolve configured/effective aggregate and per-deployment TPM/RPM.

        ``shared`` is conservative for keys belonging to one organization.
        ``per_key`` is appropriate only when each key has an independent quota.
        In both cases, ``effective_*`` is the aggregate limit handed to LMI's
        global weighted limiter; ``per_*`` is attached to LiteLLM deployments.
        """
        deployment_count = max(1, deployment_count)
        headroom = provider.rate_limit_headroom
        if not (0.0 < headroom <= 1.0):
            raise ValueError(
                f"rate_limit_headroom for provider {provider.name!r} must be in (0, 1]."
            )

        if provider.quota_scope == "per_key":
            configured_tpm = provider.per_key_tpm
            configured_rpm = provider.per_key_rpm
            per_tpm = (
                max(1, math.floor(configured_tpm * headroom))
                if configured_tpm is not None
                else None
            )
            per_rpm = (
                max(1, math.floor(configured_rpm * headroom))
                if configured_rpm is not None
                else None
            )
            effective_tpm = per_tpm * deployment_count if per_tpm is not None else None
            effective_rpm = per_rpm * deployment_count if per_rpm is not None else None
            return (
                configured_tpm,
                configured_rpm,
                effective_tpm,
                effective_rpm,
                per_tpm,
                per_rpm,
            )

        configured_tpm = provider.shared_tpm
        configured_rpm = provider.shared_rpm
        effective_tpm = (
            max(1, math.floor(configured_tpm * headroom))
            if configured_tpm is not None
            else None
        )
        effective_rpm = (
            max(1, math.floor(configured_rpm * headroom))
            if configured_rpm is not None
            else None
        )
        per_tpm = (
            max(1, effective_tpm // deployment_count)
            if effective_tpm is not None
            else None
        )
        per_rpm = (
            max(1, effective_rpm // deployment_count)
            if effective_rpm is not None
            else None
        )
        return (
            configured_tpm,
            configured_rpm,
            effective_tpm,
            effective_rpm,
            per_tpm,
            per_rpm,
        )

    @staticmethod
    def _merge_params(
        provider: ProviderSpec,
        target: ModelTargetSpec,
        *,
        model: str,
        api_key_ref: str | None = None,
        tpm: int | None = None,
        rpm: int | None = None,
    ) -> dict[str, Any]:
        params = {
            "model": model,
            "temperature": target.temperature,
            **provider.default_params,
            **target.params,
        }
        if provider.api_base:
            params["api_base"] = provider.api_base
        if api_key_ref:
            params["api_key"] = api_key_ref
        if tpm is not None:
            params["tpm"] = tpm
        if rpm is not None:
            params["rpm"] = rpm
        return params

    @abstractmethod
    def resolve(
        self,
        provider: ProviderSpec,
        target: ModelTargetSpec,
    ) -> ResolvedModelTarget:
        raise NotImplementedError
