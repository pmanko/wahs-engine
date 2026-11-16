"""Northbound & Southbound OpenAI `/v1` wire-contract tests (AC-2.3)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional

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
                context_window=32768,
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


@pytest.mark.asyncio
async def test_tool_call_batch_over_four_returns_paired_tool_messages_for_skipped_calls() -> None:
    from wahs_engine.app import create_app
    from wahs_engine.config import ModelSpec, WahsEngineSettings

    captured_requests: List[Dict[str, Any]] = []

    def llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "gemma4:12b"}]})
        body = json.loads(request.content.decode("utf-8"))
        captured_requests.append(body)
        is_round_two = any(m.get("role") == "tool" for m in body.get("messages", []))

        if body.get("stream"):
            if not is_round_two:
                tc_deltas = [
                    {
                        "index": i,
                        "id": f"call_stream_{i}",
                        "type": "function",
                        "function": {"name": "search_chart_entries", "arguments": json.dumps({"query": f"lab {i}"})},
                    }
                    for i in range(6)
                ]
                sse_lines = [
                    f'data: {json.dumps({"id": "s1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": tc_deltas}, "finish_reason": None}]})}',
                    'data: {"id":"s1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=("\n\n".join(sse_lines) + "\n\n").encode("utf-8"),
                )
            sse_lines = [
                'data: {"id":"s2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"STREAM_DONE"},"finish_reason":null}]}',
                'data: {"id":"s2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=("\n\n".join(sse_lines) + "\n\n").encode("utf-8"),
            )

        if not is_round_two:
            calls = [
                {
                    "id": f"call_sync_{i}",
                    "type": "function",
                    "function": {"name": "search_chart_entries", "arguments": json.dumps({"query": f"lab {i}"})},
                }
                for i in range(6)
            ]
            return httpx.Response(
                200,
                json={
                    "id": "sync-1",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": calls}, "finish_reason": "tool_calls"}],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "sync-2",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "SYNC_DONE"}, "finish_reason": "stop"}],
            },
        )

    settings = WahsEngineSettings(
        allow_local_bundle_mode=True,
        llm_base_url="http://mock-llm/v1",
        default_model="wahs-test",
        model_specs={
            "wahs-test": ModelSpec(
                upstream_model="gemma4:12b",
                context_window=32768,
                chart_budget_ceiling=2048,
                tokenizer_family="gemma4",
                provider_flavor="litellm",
                disable_thinking=True,
            )
        },
    )
    entries = [{"resource": {"resourceType": "Patient", "id": "p-tc6", "birthDate": "1960-01-01"}}]
    for idx in range(45):
        entries.append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-tc6-{idx}",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": f"888-{idx}", "display": f"Filler Panel Result Number {idx} with Extended Text"}]},
                    "effectiveDateTime": "2026-01-01",
                    "valueQuantity": {"value": float(idx), "unit": "mg/dL"},
                }
            }
        )
    bundle = {"resourceType": "Bundle", "type": "collection", "entry": entries}

    async with httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)) as llm_client:
        app = create_app(settings, llm_http_client=llm_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://wahs-engine"
        ) as api:
            # Streaming 6 parallel tool calls
            r_stream = await api.post(
                "/v1/chat/completions",
                json={
                    "model": "wahs-test",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Check labs"}],
                    "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 512, "enable_overflow_tools": True},
                },
            )
            assert r_stream.status_code == 200
            assert "STREAM_DONE" in r_stream.text
            stream_round2_msgs = captured_requests[1]["messages"]
            stream_asst = [m for m in stream_round2_msgs if m.get("role") == "assistant" and m.get("tool_calls")][-1]
            stream_tools = [m for m in stream_round2_msgs if m.get("role") == "tool"]
            assert [tc["id"] for tc in stream_asst["tool_calls"]] == [f"call_stream_{i}" for i in range(6)]
            assert [m["tool_call_id"] for m in stream_tools] == [f"call_stream_{i}" for i in range(6)]
            assert "tool_call_limit_exceeded" in stream_tools[4]["content"]
            assert "tool_call_limit_exceeded" in stream_tools[5]["content"]

            # Non-streaming 6 parallel tool calls
            r_sync = await api.post(
                "/v1/chat/completions",
                json={
                    "model": "wahs-test",
                    "stream": False,
                    "messages": [{"role": "user", "content": "Check labs"}],
                    "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 512, "enable_overflow_tools": True},
                },
            )
            assert r_sync.status_code == 200
            assert r_sync.json()["choices"][0]["message"]["content"] == "SYNC_DONE"
            assert len(r_sync.json()["x_wahs"]["tool_calls"]) == 4
            sync_round2_msgs = captured_requests[3]["messages"]
            sync_asst = [m for m in sync_round2_msgs if m.get("role") == "assistant" and m.get("tool_calls")][-1]
            sync_tools = [m for m in sync_round2_msgs if m.get("role") == "tool"]
            assert [tc["id"] for tc in sync_asst["tool_calls"]] == [f"call_sync_{i}" for i in range(6)]
            assert [m["tool_call_id"] for m in sync_tools] == [f"call_sync_{i}" for i in range(6)]
            assert "tool_call_limit_exceeded" in sync_tools[4]["content"]
            assert "tool_call_limit_exceeded" in sync_tools[5]["content"]


@pytest.mark.asyncio
async def test_streaming_yields_incremental_chunks_live_before_upstream_finishes_with_tools_enabled() -> None:
    from wahs_engine.config import ModelSpec

    release_tail = asyncio.Event()
    seen_contents: List[str] = []
    seen_stream_events: List[str] = []

    class _GatedUpstreamStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield (
                b'data: {"id":"live-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Your current active medications:\\n"},"finish_reason":null}]}\n\n'
            )
            await asyncio.wait_for(release_tail.wait(), timeout=3.0)
            yield (
                b'data: {"id":"live-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"- Entresto 49/51 mg"},"finish_reason":"stop"}],"usage":{"prompt_tokens":150,"completion_tokens":12,"total_tokens":162}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            pass

    async def llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "gemma-e4b"}]})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_GatedUpstreamStream(),
        )

    settings = WahsEngineSettings(
        allow_local_bundle_mode=True,
        llm_base_url="http://mock-llm/v1",
        default_model="wahs-test",
        model_specs={
            "wahs-test": ModelSpec(
                upstream_model="gemma-e4b",
                context_window=32768,
                chart_budget_ceiling=2048,
                tokenizer_family="gemma4",
                provider_flavor="llamacpp",
                disable_thinking=True,
            )
        },
    )
    entries = [{"resource": {"resourceType": "Patient", "id": "p-live", "birthDate": "1954-04-12"}}]
    for idx in range(45):
        entries.append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-live-{idx}",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": f"777-{idx}", "display": f"Filler Lab Panel {idx} Long Descriptive Name"}]},
                    "effectiveDateTime": "2026-01-01",
                    "valueQuantity": {"value": float(idx), "unit": "mg/dL"},
                }
            }
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)) as llm_client:
        app = create_app(settings, llm_http_client=llm_client)
        # Exercise the full FastAPI route + StreamingResponse directly because httpx.ASGITransport
        # buffers all `http.response.body` events until the ASGI task exits.
        req_bytes = json.dumps(
            {
                "model": "wahs-test",
                "stream": True,
                "messages": [{"role": "user", "content": "What medicines am I taking right now?"}],
                "wahs": {
                    "fhir_bundle": {"resourceType": "Bundle", "type": "collection", "entry": entries},
                    "target_budget_tokens": 512,
                    "enable_overflow_tools": True,
                },
            }
        ).encode("utf-8")
        body_sent = False
        disconnect_event = asyncio.Event()
        chunk_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

        async def _receive() -> Dict[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": req_bytes, "more_body": False}
            await disconnect_event.wait()
            return {"type": "http.disconnect"}

        async def _send(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                assert message["status"] == 200
            elif message["type"] == "http.response.body":
                if message.get("body"):
                    await chunk_queue.put(message["body"])
                if not message.get("more_body", False):
                    await chunk_queue.put(None)
                    disconnect_event.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("wahs-engine", 80),
        }

        async def _consume() -> None:
            server_task = asyncio.create_task(app(scope, _receive, _send))
            try:
                while True:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=3.0)
                    if chunk is None:
                        break
                    for raw_line in chunk.decode("utf-8").splitlines():
                        if not raw_line.startswith("data:"):
                            continue
                        payload_str = raw_line[5:].strip()
                        if payload_str == "[DONE]":
                            continue
                        obj = json.loads(payload_str)
                        xws = obj.get("x_wahs_stream") or {}
                        if xws.get("event"):
                            seen_stream_events.append(str(xws["event"]))
                        piece = ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                        if piece:
                            seen_contents.append(piece)
                            if "medications" in piece:
                                release_tail.set()
                await server_task
            finally:
                disconnect_event.set()
                if not server_task.done():
                    server_task.cancel()

        await asyncio.wait_for(_consume(), timeout=5.0)

    assert release_tail.is_set()
    assert "context_ready" in seen_stream_events
    full_text = "".join(seen_contents)
    assert "Your current active medications:\n" in full_text
    assert "- Entresto 49/51 mg" in full_text


@pytest.mark.asyncio
async def test_streaming_late_tool_call_after_multiline_draft_emits_reset_event() -> None:
    from wahs_engine.config import ModelSpec

    round_counter = {"n": 0}

    def llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "gemma-e4b"}]})
        round_counter["n"] += 1
        if round_counter["n"] == 1:
            sse_lines = [
                'data: {"id":"r1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Checking your chart now.\\nLet me verify older notes."},"finish_reason":null}]}',
                'data: {"id":"r1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_late_1","type":"function","function":{"name":"search_chart_entries","arguments":"{\\"query\\":\\"creatinine\\"}"}}]},"finish_reason":null}]}',
                'data: {"id":"r1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        else:
            sse_lines = [
                'data: {"id":"r2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"FINAL_ANSWER"},"finish_reason":null}]}',
                'data: {"id":"r2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=("\n\n".join(sse_lines) + "\n\n").encode("utf-8"),
        )

    settings = WahsEngineSettings(
        allow_local_bundle_mode=True,
        llm_base_url="http://mock-llm/v1",
        default_model="wahs-test",
        model_specs={
            "wahs-test": ModelSpec(
                upstream_model="gemma-e4b",
                context_window=32768,
                chart_budget_ceiling=2048,
                tokenizer_family="gemma4",
                provider_flavor="llamacpp",
                disable_thinking=True,
            )
        },
    )
    entries = [{"resource": {"resourceType": "Patient", "id": "p-reset", "birthDate": "1960-01-01"}}]
    for idx in range(45):
        entries.append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-reset-{idx}",
                    "status": "final",
                    "category": [{"coding": [{"code": "laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": f"666-{idx}", "display": f"Filler Lab Panel Result {idx} Long Descriptive Text"}]},
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
            stream_events: List[Dict[str, Any]] = []
            for raw_line in resp.text.splitlines():
                if not raw_line.startswith("data:"):
                    continue
                payload_str = raw_line[5:].strip()
                if payload_str == "[DONE]":
                    break
                obj = json.loads(payload_str)
                if "x_wahs_stream" in obj:
                    stream_events.append(obj["x_wahs_stream"])

    assert round_counter["n"] == 2
    tool_events = [e for e in stream_events if e.get("event") == "tool_round"]
    assert len(tool_events) == 1
    assert tool_events[0]["reset_content"] is True
    assert tool_events[0]["tools"] == ["search_chart_entries"]
    assert "FINAL_ANSWER" in resp.text


@pytest.mark.asyncio
async def test_eval_arm_and_compact_profile_resolution_all_four_pathways_and_32k_budget() -> None:
    import base64
    from wahs_engine.config import parse_wahs_profile_spec
    from wahs_engine.schemas import OpenAIChatCompletionRequest

    # 1. Pure parsing checks (harness prefixes + compact suffixes + budgets + overrides)
    p_rag = parse_wahs_profile_spec("wahs-engine-rag:ollama/gemma4:26b@16k")
    assert (p_rag.inferred_strategy, p_rag.lookup_key, p_rag.explicit_budget_tokens, p_rag.had_arm_prefix) == (
        "note_rag",
        "ollama/gemma4:26b",
        16384,
        True,
    )
    p_compact_32k = parse_wahs_profile_spec("wahs-26b-rag@32k")
    assert (
        p_compact_32k.inferred_strategy,
        p_compact_32k.lookup_key,
        p_compact_32k.explicit_budget_tokens,
        p_compact_32k.had_arm_prefix,
    ) == ("note_rag", "wahs-26b-kb", 32768, False)

    req_override = OpenAIChatCompletionRequest(
        model="wahs-26b-raw@4k",
        messages=[{"role": "user", "content": "check notes"}],
        wahs={"retrieval_strategy": "note_rag", "enable_kb": True},
    )
    resolved_ovr = req_override.resolved_wahs_options()
    assert resolved_ovr.retrieval_strategy == "note_rag"
    assert resolved_ovr.enable_kb is True
    assert resolved_ovr.target_budget_tokens == 4096

    # 2. Full ASGI HTTP tests for note_rag (@16k & @32k), tools, and chart_only
    captured_payloads: List[Dict[str, Any]] = []

    def llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": []})
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-arm-1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hold lisinopril due to potassium [1]."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 500, "completion_tokens": 12},
            },
        )

    settings = WahsEngineSettings(
        allow_local_bundle_mode=True,
        llm_base_url="http://mock-llm/v1",
        default_model="wahs-26b-kb",
        default_context_window=32768,
        default_chart_budget_ceiling=16384,
        default_provider_flavor="litellm",
        model_map={"wahs-26b-kb": "ollama/gemma4:26b"},
    )
    entries = [
        {"resource": {"resourceType": "Patient", "id": "p-arm", "birthDate": "1960-01-01"}},
        {
            "resource": {
                "resourceType": "DocumentReference",
                "id": "doc-discharge-1",
                "status": "current",
                "type": {"text": "BIDMC Discharge Summary"},
                "subject": {"reference": "Patient/p-arm"},
                "date": "2026-01-26T15:00:00Z",
                "description": "BIDMC Discharge Summary",
                "content": [
                    {
                        "attachment": {
                            "contentType": "text/plain; charset=utf-8",
                            "data": base64.b64encode(
                                b"Hold lisinopril immediately due to hyperkalemia potassium 5.6 mmol/L."
                            ).decode("ascii"),
                        }
                    }
                ],
            }
        },
    ]
    bundle = {"resourceType": "Bundle", "type": "collection", "entry": entries}

    async with httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)) as llm_client:
        app = create_app(settings, llm_http_client=llm_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://wahs-engine"
        ) as api:
            r_models = await api.get("/v1/models")
            mids = [m["id"] for m in r_models.json()["data"]]
            assert "wahs-26b-rag" in mids
            assert "wahs-26b-tools" in mids
            assert "wahs-26b-chart" in mids

            # A. Note-RAG via compact profile + @32k unclamping
            r_rag32 = await api.post(
                "/v1/chat/completions",
                json={
                    "model": "wahs-26b-rag@32k",
                    "stream": False,
                    "messages": [{"role": "user", "content": "Why hold lisinopril potassium?"}],
                    "wahs": {"fhir_bundle": bundle},
                },
            )
            assert r_rag32.status_code == 200
            xw_rag32 = r_rag32.json()["x_wahs"]
            assert xw_rag32["retrieval_strategy"] == "note_rag"
            assert xw_rag32["ai_provenance"]["recorded_model"] == "ollama/gemma4:26b"
            assert xw_rag32["condensation"]["context_window"] == 65536
            assert xw_rag32["condensation"]["budget_tokens"] == 32768
            assert xw_rag32["note_rag"]["passages_retrieved"] >= 1
            assert captured_payloads[-1]["model"] == "ollama/gemma4:26b"
            assert captured_payloads[-1]["num_ctx"] == 65536

            # B. Verbatim harness eval arm ID (`wahs-engine-rag:gemma-26b@16k` cross-provider mapped)
            r_harness = await api.post(
                "/v1/chat/completions",
                json={
                    "model": "wahs-engine-rag:gemma-26b@16k",
                    "stream": False,
                    "messages": [{"role": "user", "content": "Potassium discharge instructions?"}],
                    "wahs": {"fhir_bundle": bundle},
                },
            )
            assert r_harness.status_code == 200
            xw_har = r_harness.json()["x_wahs"]
            assert xw_har["retrieval_strategy"] == "note_rag"
            assert xw_har["condensation"]["budget_tokens"] == 16384
            assert captured_payloads[-1]["model"] == "ollama/gemma4:26b"

            # C. Un-suffixed model + huge caller target_budget_tokens STILL clamps to 16384
            r_clamped = await api.post(
                "/v1/wahs/context/preview",
                json={
                    "model": "wahs-26b-kb",
                    "wahs": {"fhir_bundle": bundle, "target_budget_tokens": 32768},
                },
            )
            assert r_clamped.status_code == 200
            assert r_clamped.json()["x_wahs"]["condensation"]["clamp_reason"] == "clamped_to_chart_budget_ceiling"
            assert r_clamped.json()["x_wahs"]["condensation"]["budget_effective"] == 16384

            # D. Dynamic FHIR Tools
            r_tools = await api.post(
                "/v1/wahs/context/preview",
                json={"model": "wahs-26b-tools", "wahs": {"fhir_bundle": bundle}},
            )
            assert r_tools.status_code == 200
            assert r_tools.json()["x_wahs"]["retrieval_strategy"] == "dynamic_fhir_tools"
            assert r_tools.json()["x_wahs"]["condensation"]["budget_effective"] == 2048

            # E. Chart Only (`-raw` and `-chart`) disables KB cleanly
            for chart_model in ("wahs-26b-raw", "wahs-26b-chart"):
                r_raw = await api.post(
                    "/v1/wahs/context/preview",
                    json={"model": chart_model, "wahs": {"fhir_bundle": bundle}},
                )
                assert r_raw.status_code == 200
                xw_raw = r_raw.json()["x_wahs"]
                assert xw_raw["retrieval_strategy"] == "chart_only"
                assert xw_raw["ai_provenance"]["kb_enabled"] is False
