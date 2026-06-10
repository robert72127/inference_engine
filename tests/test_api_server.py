import json
import unittest

from fastapi import HTTPException

import api_server
from api_server import ChatCompletionRequest, ChatMessage, app, chat, validate_sampling_params
from models import MODEL


class FakeEngine:
    def __init__(self):
        self.model = MODEL.QWEN_2_5_0_5B_INSTRUCT
        self.generate_calls = []
        self.generate_stream_calls = []

    async def generate(self, prompt, max_tokens, temperature=1.0, top_p=1.0, top_k=None):
        self.generate_calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
        return "ok", 5, 3

    async def generate_stream(self, prompt, max_tokens, temperature=1.0, top_p=1.0, top_k=None):
        self.generate_stream_calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
        from engine import GenStart, GenDelta, GenEnd

        yield GenStart(prompt_tokens=5)

        for chunk in ("o", "k"):
            yield GenDelta(text=chunk)

        yield GenEnd(completion_tokens=2)


class ApiServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_engine = getattr(app.state, "engine", None)
        app.state.engine = FakeEngine()

    async def asyncTearDown(self):
        if self.original_engine is None:
            try:
                del app.state.engine
            except AttributeError:
                pass
        else:
            app.state.engine = self.original_engine

    async def test_chat_passes_sampling_params_to_generate(self):
        req = ChatCompletionRequest(
            model=MODEL.QWEN_2_5_0_5B_INSTRUCT.value,
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=7,
            temperature=0.4,
            top_p=0.8,
            stream=False,
        )

        response = await chat(req)

        self.assertEqual(response["choices"][0]["message"]["content"], "ok")
        self.assertEqual(
            app.state.engine.generate_calls,
            [
                {
                    "prompt": "hello",
                    "max_tokens": 7,
                    "temperature": 0.4,
                    "top_p": 0.8,
                    "top_k": None,
                }
            ],
        )

    async def test_chat_stream_passes_sampling_params_to_generate_stream(self):
        req = ChatCompletionRequest(
            model=MODEL.QWEN_2_5_0_5B_INSTRUCT.value,
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=5,
            temperature=0.9,
            top_p=0.7,
            stream=True,
        )

        response = await chat(req)
        body = []
        async for chunk in response.body_iterator:
            body.append(chunk if isinstance(chunk, str) else chunk.decode())
        payload = "".join(body)

        self.assertIn('"content": "o"', payload)
        self.assertIn('"content": "k"', payload)
        self.assertEqual(
            app.state.engine.generate_stream_calls,
            [
                {
                    "prompt": "hello",
                    "max_tokens": 5,
                    "temperature": 0.9,
                    "top_p": 0.7,
                    "top_k": None,
                }
            ],
        )

    async def test_chat_passes_top_k_to_generate(self):
        req = ChatCompletionRequest(
            model=MODEL.QWEN_2_5_0_5B_INSTRUCT.value,
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=5,
            top_k=10,
            stream=False,
        )

        await chat(req)

        self.assertEqual(app.state.engine.generate_calls[0]["top_k"], 10)

    async def test_chat_stream_passes_top_k_to_generate_stream(self):
        req = ChatCompletionRequest(
            model=MODEL.QWEN_2_5_0_5B_INSTRUCT.value,
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=5,
            top_k=20,
            stream=True,
        )

        response = await chat(req)
        async for _ in response.body_iterator:
            pass

        self.assertEqual(app.state.engine.generate_stream_calls[0]["top_k"], 20)

    async def test_chat_rejects_invalid_sampling_params(self):
        req = ChatCompletionRequest(
            model=MODEL.QWEN_2_5_0_5B_INSTRUCT.value,
            messages=[ChatMessage(role="user", content="hello")],
            temperature=-0.1,
            top_p=1.0,
        )

        with self.assertRaises(HTTPException) as ctx:
            await chat(req)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("temperature", ctx.exception.detail)


class ValidateSamplingParamsTests(unittest.TestCase):
    def test_accepts_openai_style_values(self):
        validate_sampling_params(0.0, 1.0, None)
        validate_sampling_params(1.0, 0.5, None)

    def test_rejects_invalid_top_p(self):
        with self.assertRaises(HTTPException):
            validate_sampling_params(1.0, 0.0, None)

        with self.assertRaises(HTTPException):
            validate_sampling_params(1.0, 1.1, None)

    def test_accepts_valid_top_k(self):
        validate_sampling_params(1.0, 1.0, 1)
        validate_sampling_params(1.0, 1.0, 50)

    def test_accepts_none_top_k(self):
        validate_sampling_params(1.0, 1.0, None)

    def test_rejects_top_k_zero(self):
        with self.assertRaises(HTTPException) as ctx:
            validate_sampling_params(1.0, 1.0, 0)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("top_k", ctx.exception.detail)

    def test_rejects_negative_top_k(self):
        with self.assertRaises(HTTPException) as ctx:
            validate_sampling_params(1.0, 1.0, -1)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("top_k", ctx.exception.detail)
