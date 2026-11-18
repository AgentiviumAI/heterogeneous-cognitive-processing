from __future__ import annotations

from .base import ModelTargetSpec, ProviderAdapter, ProviderSpec, ResolvedModelTarget


def _with_prefix(model: str, prefix: str | None) -> str:
    if not prefix:
        return model
    if model.startswith(f"{prefix}/"):
        return model
    return f"{prefix}/{model}"


def _resolve_common(
    adapter: ProviderAdapter,
    provider: ProviderSpec,
    target: ModelTargetSpec,
    *,
    model: str,
) -> ResolvedModelTarget:
    adapter.preflight(provider)
    key_refs = adapter._api_key_references(provider)

    # Providers without auth (e.g. a local vLLM) still need one deployment.
    deployment_count = len(key_refs) or 1
    (
        configured_tpm,
        configured_rpm,
        effective_tpm,
        effective_rpm,
        per_tpm,
        per_rpm,
    ) = adapter._rate_limits(provider, deployment_count)

    refs: tuple[str | None, ...] = key_refs if key_refs else (None,)
    deployments = tuple(
        {
            "model_name": model,
            "litellm_params": adapter._merge_params(
                provider,
                target,
                model=model,
                api_key_ref=key_ref,
                tpm=per_tpm,
                rpm=per_rpm,
            ),
        }
        for key_ref in refs
    )
    first_params = dict(deployments[0]["litellm_params"])

    return ResolvedModelTarget(
        target_name=target.name,
        provider_name=provider.name,
        adapter_name=adapter.name,
        requested_model=target.model,
        litellm_model=model,
        temperature=target.temperature,
        litellm_params=first_params,
        router_kwargs=dict(provider.router_kwargs),
        api_key_env=provider.api_key_env,
        api_keys_env=provider.api_keys_env,
        api_base=provider.api_base,
        deployments=deployments,
        key_count=len(key_refs),
        quota_scope=provider.quota_scope,
        configured_tpm=configured_tpm,
        configured_rpm=configured_rpm,
        effective_tpm=effective_tpm,
        effective_rpm=effective_rpm,
        per_deployment_tpm=per_tpm,
        per_deployment_rpm=per_rpm,
        rate_limit_headroom=provider.rate_limit_headroom,
        adaptive_rate_limit_recovery=provider.adaptive_rate_limit_recovery,
        rate_limit_recovery_attempts=provider.rate_limit_recovery_attempts,
        rate_limit_default_wait_seconds=provider.rate_limit_default_wait_seconds,
        rate_limit_max_wait_seconds=provider.rate_limit_max_wait_seconds,
    )


class LiteLLMNativeAdapter(ProviderAdapter):
    """Use a provider supported natively by LiteLLM."""

    name = "litellm"

    def resolve(
        self,
        provider: ProviderSpec,
        target: ModelTargetSpec,
    ) -> ResolvedModelTarget:
        model = _with_prefix(target.model, provider.litellm_prefix)
        return _resolve_common(self, provider, target, model=model)


class OpenAICompatibleAdapter(ProviderAdapter):
    """Route an OpenAI-compatible endpoint through LiteLLM's OpenAI backend."""

    name = "openai_compatible"

    def resolve(
        self,
        provider: ProviderSpec,
        target: ModelTargetSpec,
    ) -> ResolvedModelTarget:
        if not provider.api_base:
            raise ValueError(
                f"Provider {provider.name!r} uses openai_compatible but has no api_base."
            )
        model = _with_prefix(target.model, provider.litellm_prefix or "openai")
        return _resolve_common(self, provider, target, model=model)
