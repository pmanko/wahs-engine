"""Regression tests for PR #10 audit findings (E1, E2, E3, E10)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from wahs_engine.assembler import WAHSBundleAssembler
from wahs_engine.config import ModelSpec, WahsEngineSettings
from wahs_engine.jwt_claims import VerifiedCallerIdentity
from wahs_engine.schemas import OpenAIChatCompletionRequest, WahsRequestOptions


def test_e1_single_flight_survives_first_caller_timeout() -> None:
    """E1: Request 1 timing out must not close the shared httpx.AsyncClient under Request 2."""

    async def _run() -> None:
        everything_calls = 0
        release_everything = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal everything_calls
            path = request.url.path
            if path.endswith("/Patient/master-1"):
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Patient",
                        "id": "master-1",
                        "name": [{"given": ["Maurice"], "family": "Auer"}],
                        "meta": {"lastUpdated": "2026-11-19T12:00:00Z"},
                        "link": [{"other": {"reference": "Patient/ds-1"}}],
                    },
                )
            if path.endswith("/Patient/ds-1"):
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Patient",
                        "id": "ds-1",
                        "meta": {
                            "lastUpdated": "2026-11-19T12:00:00Z",
                            "tag": [
                                {
                                    "system": "http://fhir.cirg.washington.edu/CodeSystem/wahealthsummary",
                                    "code": "ehr|fhir|uw",
                                }
                            ],
                        },
                    },
                )
            if path.endswith("/Patient/ds-1/$everything"):
                everything_calls += 1
                await release_everything.wait()
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Bundle",
                        "type": "searchset",
                        "entry": [
                            {
                                "resource": {
                                    "resourceType": "Condition",
                                    "id": "cond-1",
                                    "clinicalStatus": {"coding": [{"code": "active"}]},
                                    "code": {"text": "Hypertension"},
                                    "subject": {"reference": "Patient/ds-1"},
                                }
                            }
                        ],
                    },
                )
            return httpx.Response(404, json={"resourceType": "OperationOutcome"})

        transport = httpx.MockTransport(handler)
        settings = WahsEngineSettings(
            fhir_auth_base_url="http://fhir-auth:8080/fhir",
            assembly_timeout_seconds=10.0,
        )
        assembler = WAHSBundleAssembler(settings)
        assembler._shared_client = httpx.AsyncClient(transport=transport, timeout=5.0)

        ident = VerifiedCallerIdentity(
            sub="sub-1", raw_token="tok-1", token_fingerprint="fp-1", claims={"sub": "sub-1"}
        )
        opts = WahsRequestOptions(master_patient_id="master-1")

        req1 = asyncio.create_task(
            asyncio.wait_for(
                assembler.assemble(options=opts, identity=ident, requested_model="wahs-26b-kb"),
                timeout=0.05,
            )
        )
        await asyncio.sleep(0.01)
        req2 = asyncio.create_task(
            assembler.assemble(options=opts, identity=ident, requested_model="wahs-26b-kb")
        )

        with pytest.raises(asyncio.TimeoutError):
            await req1

        assert not assembler._shared_client.is_closed
        release_everything.set()
        ctx2 = await req2
        assert ctx2.patient_id == "master-1"
        assert everything_calls == 1
        await assembler.aclose()
        assert assembler._shared_client is None

    asyncio.run(_run())


def test_e3_forced_refresh_updates_dataset_cache_for_subsequent_scopes() -> None:
    """E3: `refresh=True` must update `_dataset_cache` so a later turn with new scope sees fresh data."""

    async def _run() -> None:
        current_condition_text = "Stage 1Initial"

        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/Patient/master-1"):
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Patient",
                        "id": "master-1",
                        "meta": {"lastUpdated": "2026-11-19T12:00:00Z"},
                        "link": [{"other": {"reference": "Patient/ds-1"}}],
                    },
                )
            if path.endswith("/Patient/ds-1"):
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Patient",
                        "id": "ds-1",
                        "meta": {"lastUpdated": "2026-11-19T12:00:00Z"},
                    },
                )
            if path.endswith("/Patient/ds-1/$everything"):
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Bundle",
                        "type": "searchset",
                        "entry": [
                            {
                                "resource": {
                                    "resourceType": "Condition",
                                    "id": "cond-inplace",
                                    "clinicalStatus": {"coding": [{"code": "active"}]},
                                    "code": {"text": current_condition_text},
                                    "subject": {"reference": "Patient/ds-1"},
                                }
                            }
                        ],
                    },
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        assembler = WAHSBundleAssembler(WahsEngineSettings(), http_client=client)
        ident = VerifiedCallerIdentity(
            sub="sub-1", raw_token="tok-1", token_fingerprint="fp-1", claims={"sub": "sub-1"}
        )

        ctx_initial = await assembler.assemble(
            options=WahsRequestOptions(master_patient_id="master-1"),
            identity=ident,
        )
        assert "Stage 1Initial" in ctx_initial.condensed.chart_text

        current_condition_text = "Stage 2UpdatedInPlace"
        ctx_refreshed = await assembler.assemble(
            options=WahsRequestOptions(master_patient_id="master-1", refresh=True),
            identity=ident,
        )
        assert "Stage 2UpdatedInPlace" in ctx_refreshed.condensed.chart_text

        ctx_next_scope = await assembler.assemble(
            options=WahsRequestOptions(master_patient_id="master-1", target_budget_tokens=4096),
            identity=ident,
        )
        assert ctx_next_scope.dataset_cache_hits == 1
        assert "Stage 2UpdatedInPlace" in ctx_next_scope.condensed.chart_text
        await client.aclose()

    asyncio.run(_run())


def test_e2_and_e10_budget_ceiling_cross_key_tokenizer_and_default_model_options() -> None:
    """E2 + E10: Cap `@<budget>` context lift, map cross_key specs & non-Gemma tokenizers, and honor default_model."""
    settings = WahsEngineSettings(
        default_model="wahs-26b-rag@16k",
        default_context_window=32768,
        max_context_window=131072,
        default_chart_budget_ceiling=16384,
        model_map={"wahs-26b-kb": "ollama/gemma4:26b", "wahs-qwen3-14b-kb": "ollama/qwen3:14b"},
        model_specs={
            "wahs-26b-kb": ModelSpec(
                upstream_model="ollama/gemma4:26b",
                context_window=49152,
                chart_budget_ceiling=24576,
                tokenizer_family="gemma4",
            )
        },
    )

    huge_spec = settings.resolve_model_spec("wahs-26b-kb@500000k")
    assert huge_spec.context_window == 131072
    assert huge_spec.chart_budget_ceiling == 65536

    cal_spec = settings.resolve_model_spec("wahs-26b-kb@32k")
    assert cal_spec.context_window == 65536
    assert cal_spec.chart_budget_ceiling == 32768

    cross_spec = settings.resolve_model_spec("ollama/gemma4:26b")
    assert cross_spec.context_window == 49152
    assert cross_spec.chart_budget_ceiling == 24576

    qwen_spec = settings.resolve_model_spec("wahs-qwen3-14b-kb")
    assert qwen_spec.tokenizer_family == "o200k_base"
    gemma_spec = settings.resolve_model_spec("wahs-12b-kb")
    assert gemma_spec.tokenizer_family == "gemma4"

    req = OpenAIChatCompletionRequest(
        model=None,
        messages=[{"role": "user", "content": "Hello"}],
    )
    opts = req.resolved_wahs_options(default_model=settings.default_model)
    assert opts.retrieval_strategy == "note_rag"
    assert opts.target_budget_tokens == 16384
