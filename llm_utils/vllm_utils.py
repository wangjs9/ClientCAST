"""
This is a file to utilize the local API models, implemented by vllm.
"""
import json
import logging
import os
import time
from typing import List, Union
import uuid
import re
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential, stop_never
from groq import Groq
from openai import OpenAI
from together import Together
from octoai.text_gen import ChatMessage
from octoai.client import OctoAI

from chatarena.message import Message
from chatarena.backends import IntelligenceBackend, register_backend

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
SYSTEM_NAME = "System"
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class VLLMInteract(IntelligenceBackend):
    stateful = False
    type_name = "vllm-interact"
    headers = {'Content-Type': 'application/json'}

    def __init__(
            self,
            temperature: float = DEFAULT_TEMPERATURE,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            model: str = None,
            model_url: str = None,
            merge_other_agents_as_one_user: bool = True,
            **kwargs,
    ):
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            model_url=model_url,
            merge_other_agents_as_one_user=merge_other_agents_as_one_user,
            **kwargs
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.model_url = model_url
        self.merge_other_agent_as_user = merge_other_agents_as_one_user
        self.system_name = "system" if "LLaMA" in self.model else "user"

    # @retry(stop=stop_after_attempt(12), wait=wait_random_exponential(min=60, max=120))
    @retry(stop=stop_never, wait=wait_random_exponential(multiplier=60, exp_base=2, min=30, max=120))
    def _get_response(self, messages):
        json_data = {
            'model': self.model,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        completion = requests.post(self.model_url, headers=self.headers, json=json_data).json()
        response = completion["choices"][0]["message"]["content"]
        response = response.strip()
        # logging.info("========================")
        # logging.info(completion)
        # logging.info(response)
        return response

    def query(
            self,
            agent_name: str,
            role_desc: str,
            history_messages: List[Message],
            global_prompt: str = None,
            request_msg: Message = None,
            *args,
            **kwargs,
    ) -> str:
        # Merge the role description and the global prompt as the system prompt for the agent
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt = f"{global_prompt.strip()}\n\n{role_desc}"
        else:
            system_prompt = role_desc

        all_messages = [(SYSTEM_NAME, system_prompt)]
        total_words = 5300 - len(system_prompt.split())
        # total_words = 3800 - len(system_prompt.split())
        # logging.info("Total word in the conversation: %d" % (total_words))
        history_messages_len = [len(msg.content.split()) + 2 for msg in history_messages]
        if sum(history_messages_len) > total_words:
            current_total = 0
            for i in range(len(history_messages_len) - 1, -1, -1):
                current_total += history_messages_len[i]
                if current_total > total_words:
                    history_messages = history_messages[i + 1:]
                    break

        for msg in history_messages:
            if msg.agent_name == SYSTEM_NAME:
                all_messages.append((SYSTEM_NAME, msg.content))
            else:  # non-system messages are suffixed with the end of message token
                all_messages.append((msg.agent_name, f"{msg.content}"))

        if request_msg:
            all_messages.append((SYSTEM_NAME, request_msg.content))
        else:  # The default request message that reminds the agent its role and instruct it to speak
            all_messages.append(
                (SYSTEM_NAME, f"Now you speak, {agent_name}.")
            )

        last_agent_idx = len(all_messages)
        for i in range(len(all_messages) - 1, 1, -1):
            if all_messages[i][0] == agent_name:
                last_agent_idx = i
                break

        messages = []
        for i, msg in enumerate(all_messages[:last_agent_idx]):
            if i == 0:
                assert (
                        msg[0] == SYSTEM_NAME
                )  # The first message should be from the system
                messages.append({"role": self.system_name, "content": msg[1]})
            else:
                if msg[0] == SYSTEM_NAME:
                    if messages[-1]["role"] == "user":
                        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{msg[1]}"
                elif messages[-1]["role"] == "user" or messages[-1]["role"] == "assistant":
                    messages[-1]["content"] = f"{messages[-1]['content']}\n[{msg[0]}]: {msg[1]}"
                elif messages[-1]["role"] == "system":
                    messages.append(
                        {"role": "user", "content": f"[{msg[0]}]: {msg[1]}"}
                    )
                else:
                    raise ValueError(f"Invalid role: {messages[-1]['role']}")
        if len(messages) > 1:
            messages[1]["content"] = "The counseling history is:\n" + messages[1]["content"]
        if last_agent_idx < len(all_messages):
            for i, msg in enumerate(all_messages[last_agent_idx:]):
                if i == 0:
                    messages.append({"role": "assistant", "content": f"[{msg[0]}]: {msg[1]}"})
                elif i == 1:
                    messages.append({"role": "user", "content": f"[{msg[0]}]: {msg[1]}"})
                else:
                    if msg[0] == SYSTEM_NAME:
                        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{msg[1]}"
                    else:
                        messages[-1]["content"] = f"{messages[-1]['content']}\n[{msg[0]}]: {msg[1]}"
        # messages[1]["content"] = "The session starts:\n" + messages[1]["content"]
        response = self._get_response(messages, *args, **kwargs)

        # Remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[.*]:", "", response).strip()  # noqa: F541
        response = re.sub(
            rf"^\s*{re.escape(agent_name)}\s*:", "", response, 1
        ).strip()  # noqa: F451

        return response
