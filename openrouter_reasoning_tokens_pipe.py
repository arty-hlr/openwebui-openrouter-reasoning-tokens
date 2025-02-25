"""
title: OpenRouter Reasoning Tokens
author: rmarfil3
repo: https://github.com/rmarfil3/openwebui-openrouter-reasoning-tokens
date: 2025-01-30
version: 0.2
license: MIT
description: Enables reasoning tokens and displays them visually in OWUI

NOTE:
After installing and enabling, new models will be added which will have the "Thinking..." feature.
These models will be prefixed with "reasoning/".

You will probably have an issue with the generated titles.
You can go to Admin Panel -> Settings -> Interface, then change the Task Model to something else instead of "Current Model".

UPDATES:
v0.2:
- Asynchronous API call to avoid blocking the UI
- Fixed reasoning responses showing up as actual response
"""

from pydantic import BaseModel, Field
import httpx
import json
import time


class Pipe:
    class Valves(BaseModel):
        OPENROUTER_API_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
        OPENROUTER_API_KEY: str = Field(
            default="",
            description="Your OpenRouter API key for authentication",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.timeout = 60

    def pipes(self):
        models = [
            "deepseek/deepseek-r1-distill-llama-70b",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-r1:free",
            "anthropic/claude-3.7-sonnet"
        ]

        return [
            {
                "id": f"reasoning/{model}",
                "name": f"reasoning/{model}",
            }
            for model in models
        ]

    async def pipe(self, body: dict):
        modified_body = {**body}
        if "model" in modified_body:
            # Remove "reasoning/" prefix from model ID
            modified_body["model"] = (
                modified_body["model"].split(".", 1)[-1].replace("reasoning/", "", 1)
            )

        modified_body["include_reasoning"] = True

        headers = {
            "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openwebui.com/",
            "X-Title": "Open WebUI",
        }

        try:
            if body.get("stream", False):
                return self._handle_streaming_request(modified_body, headers)
            else:
                return self._handle_normal_request(modified_body, headers)
        except Exception as e:
            print(e)
            return json.dumps({"error": str(e)})

    async def _handle_normal_request(self, body: dict, headers: dict):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.valves.OPENROUTER_API_BASE_URL}/chat/completions",
                json=body,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            if "choices" in data:
                for choice in data["choices"]:
                    if "message" in choice and "reasoning" in choice["message"]:
                        reasoning = choice["message"]["reasoning"]
                        choice["message"][
                            "content"
                        ] = f"<think>{reasoning}</think>\n{choice['message']['content']}"
            return data

    async def _handle_streaming_request(self, body: dict, headers: dict):
        def construct_chunk(content: str):
            return f"""data: {json.dumps({
                'id': data.get('id'),
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': body.get("model"),
                'choices': [{
                    'index': 0,
                    'delta': {'content': content, 'role': 'assistant'},
                    'finish_reason': None
                }]
            })}\n\n"""

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.valves.OPENROUTER_API_BASE_URL}/chat/completions",
                json=body,
                headers=headers,
            ) as response:
                response.raise_for_status()  # Fail fast if needed

                thinking_state = -1  # -1: not started, 0: thinking, 1: answered

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = json.loads(line[6:])
                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})

                    # State transitions
                    if thinking_state == -1 and delta.get("reasoning"):
                        thinking_state = 0
                        yield construct_chunk("<think>")

                    elif thinking_state == 0 and not delta.get("reasoning") and delta.get("content"):
                        thinking_state = 1
                        yield construct_chunk("</think>\n\n")

                    # Handle content output
                    content = delta.get("reasoning") or delta.get("content", "")
                    if content:
                        yield construct_chunk(content)

                    if choice.get("finish_reason"):
                        yield "data: [DONE]\n\n"
                        return
