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
