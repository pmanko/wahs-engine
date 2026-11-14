"""Northbound & Southbound OpenAI `/v1` wire-contract tests (AC-2.3)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from tests.test_assembler_and_security import MockFhirAuthServer, TEST_SECRET
from wahs_engine.app import create_app
from wahs_engine.config import WahsEngineSettings


def _build_mock_llm_client(call_log: List[Dict[str, Any]]) -> httpx.AsyncClient:
    def llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "ollama/gemma4:12b", "object": "model"}]},
            )

        body = json.loads(request.content.decode("utf-8"))
        call_log.append(body)

        if body.get("stream"):
            # Stream `<think>` split across chunks to verify _StreamingThinkStripper + SSE compatibility
            sse_lines = [
                'data: {"id":"up-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"<thi"},"finish_reason":null}]}\n\n',
                'data: {"id":"up-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"nk>internal chain</think>Your active medication is "},"finish_reason":null}]}\n\n',
                'data: {"id":"up-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Lisinopril 10 mg [3], and you noted a cough [5]."},"finish_reason":"stop"}]}\n\n',
                "data: [DONE]\n\n",
            ]
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content="".join(sse_lines).encode("utf-8"),
            )

        # Non-streaming: if tools were offered on turn 1, exercise single-round Tier-0 overflow tool call!
        has_tool_reply = any(m.get("role") == "tool" for m in body.get("messages", []))
        if body.get("tools") and not has_tool_reply:
            return httpx.Response(
                200,
                json={
                    "id": "up-tool-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_creatinine",
                                        "type": "function",
                                        "function": {
                                            "name": "get_lab_history",
                                            "arguments": json.dumps({"analyte": "Creatinine", "limit": 5}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )

        return httpx.Response(
            200,
            json={
                "id": "up-final-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "<think>hidden</think>Creatinine was 1.35 mg/dL on 2026-10-20 [4], and Lisinopril is active [3].",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 520, "completion_tokens": 24, "total_tokens": 544},
            },
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(llm_handler))


@pytest.mark.asyncio
async def test_openai_models_streaming_sse_vanilla_caller_and_overflow_tool_loop() -> None:
    fhir_mock = MockFhirAuthServer()
    llm_calls: List[Dict[str, Any]] = []
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(fhir_mock.handler)) as fhir_client,
        _build_mock_llm_client(llm_calls) as llm_client,
    ):
        settings = WahsEngineSettings(
            fhir_auth_base_url="http://fhir-auth:8080/fhir",
            llm_base_url="http://gpu-backend:8000/v1",
            default_model="wahs-12b-kb",
            model_map={"wahs-12b-kb": "ollama/gemma4:12b"},
            jwt_hs256_secret=TEST_SECRET,
        )
        app = create_app(settings, fhir_http_client=fhir_client, llm_http_client=llm_client)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://wahs-engine"
        ) as api:
            # 1. Health & /v1/models
            r_health = await api.get("/healthz")
            assert r_health.status_code == 200
            assert r_health.json()["status"] == "ok"

            r_models = await api.get("/v1/models")
            assert r_models.status_code == 200
            model_ids = {m["id"] for m in r_models.json()["data"]}
            assert "wahs-12b-kb" in model_ids
            assert "ollama/gemma4:12b" in model_ids

            alice_headers = {"Authorization": f"Bearer {fhir_mock.valid_tokens['user-alice']}"}

            # 2. Vanilla OpenAI streaming request (NO `wahs` field, exact `@ai-sdk/openai-compatible` shape)
            r_stream = await api.post(
                "/v1/chat/completions",
                headers=alice_headers,
                json={
                    "model": "wahs-12b-kb",
                    "stream": True,
                    "messages": [{"role": "user", "content": "What medications am I taking?"}],
                },
            )
            assert r_stream.status_code == 200
            assert r_stream.headers["content-type"].startswith("text/event-stream")
            assert r_stream.headers.get("X-WAHS-Prefix-Hash")

            chunks = []
            saw_done = False
            for line in r_stream.text.splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    saw_done = True
                    continue
                chunks.append(json.loads(payload))

            assert saw_done is True
            full_text = "".join(
                (c["choices"][0].get("delta") or {}).get("content") or "" for c in chunks
            )
            assert "<think>" not in full_text
            assert "internal chain" not in full_text
            assert "Your active medication is Lisinopril 10 mg [3]" in full_text

            final_chunk = chunks[-1]
            assert final_chunk["choices"][0]["finish_reason"] == "stop"
            x_wahs = final_chunk["x_wahs"]
            assert x_wahs["citation_map"]
            assert x_wahs["ai_provenance"]["profile"].endswith("/AI-Provenance")
            assert x_wahs["ai_provenance"]["recorded_model"] == "ollama/gemma4:12b"
            assert llm_calls[0]["max_tokens"] == 1024
            assert llm_calls[-1]["model"] == "ollama/gemma4:12b"

            # 3. Tight-budget non-streaming request (budget=512 -> omitted_clinical_items > 0 -> exercises Tier-0 tool loop!)
            r_sync = await api.post(
                "/v1/chat/completions",
                headers=alice_headers,
                json={
                    "model": "wahs-12b-kb",
                    "stream": False,
                    "messages": [{"role": "user", "content": "What was my latest creatinine?"}],
                    "wahs": {"target_budget_tokens": 512},
                },
            )
            assert r_sync.status_code == 200
            sync_body = r_sync.json()
            assert "<think>" not in sync_body["choices"][0]["message"]["content"]
            assert "Creatinine was 1.35 mg/dL" in sync_body["choices"][0]["message"]["content"]
            assert sync_body["x_wahs"]["condensation"]["budget_tokens"] == 512
            assert sync_body["x_wahs"]["condensation"]["actual_tokens"] <= 512
            assert sync_body["x_wahs"]["condensation"]["overflow_recovered"] >= 1


@pytest.mark.asyncio
async def test_streaming_buffers_pre_tool_round_text_until_terminal_round() -> None:
    from wahs_engine.app import create_app
    from wahs_engine.config import ModelSpec, WahsEngineSettings

    round_counter = {"n": 0}

    def llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "gemma4:12b"}]})
        round_counter["n"] += 1
        if round_counter["n"] == 1:
            sse_lines = [
                'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"PRE_TOOL_DRAFT "},"finish_reason":null}]}',
                'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search_chart_entries","arguments":"{\\"query\\":\\"creatinine\\"}"}}]},"finish_reason":null}]}',
                'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}',
                "data: [DONE]",
            ]
        else:
            sse_lines = [
                'data: {"id":"c2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"FINAL_ANSWER"},"finish_reason":null}]}',
                'data: {"id":"c2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":5,"total_tokens":125}}',
                "data: [DONE]",
            ]
        body = ("\n\n".join(sse_lines) + "\n\n").encode("utf-8")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    settings = WahsEngineSettings(
        allow_local_bundle_mode=True,
        llm_base_url="http://mock-llm/v1",
        default_model="wahs-test",
        model_specs={
            "wahs-test": ModelSpec(
                upstream_model="gemma4:12b",
                context_window=8192,
                chart_budget_ceiling=2048,
                tokenizer_family="gemma4",
                provider_flavor="litellm",
                disable_thinking=True,
            )
        },
    )
    entries = [{"resource": {"resourceType": "Patient", "id": "p-stream", "birthDate": "1960-01-01"}}]
    for idx in range(45):
        entries.append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-fill-{idx}",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": f"999-{idx}", "display": f"Filler Lab Panel Result Number {idx} with Extended Descriptive Text"}]},
                    "effectiveDateTime": "2026-01-01",
                    "valueQuantity": {"value": float(idx), "unit": "mg/dL"},
                }
            }
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)) as llm_client:
        app = create_app(settings, llm_http_client=llm_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://wahs-engine"
        ) as api:
            resp = await api.post(
                "/v1/chat/completions",
                json={
                    "model": "wahs-test",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Check labs"}],
                    "wahs": {
                        "fhir_bundle": {"resourceType": "Bundle", "type": "collection", "entry": entries},
                        "target_budget_tokens": 512,
                        "enable_overflow_tools": True,
                    },
                },
            )
            assert resp.status_code == 200
            raw_sse = resp.text

    assert round_counter["n"] == 2
    assert "PRE_TOOL_DRAFT" not in raw_sse
    assert "FINAL_ANSWER" in raw_sse
