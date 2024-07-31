"""
This is a file to utilize the open-source models
"""
from typing import List, Union
import re
import uuid

import torch

from chatarena.message import Message
from chatarena.backends import TransformersConversational
from transformers.pipelines.conversational import Conversation

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "mistralai/Mixtral-8x7B-v0.1"

SYSTEM_NAME = "System"
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class OpenLLMChat(TransformersConversational):
    """Interface to the open-source models with system, user, assitant roles separation."""
    stateful = False
    type_name = "opensource-chat"

    def __init__(
            self,
            temperature: float = DEFAULT_TEMPERATURE,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            model: str = DEFAULT_MODEL,
            device: int = -1,
            **kwargs,
    ):
        """Instantiate the OpenLLMChat backend.
        args:
            temperature: the temperature of the sampling
            max_tokens: the maximum number of tokens to sample
            model: the model to use
            merge_other_agents_as_one_user: whether to merge messages from other agents as one user message
        """
        kwargs = kwargs if kwargs else {}
        kwargs["max_tokens"] = max_tokens
        if temperature > 0:
            kwargs["temperature"] = temperature

        super().__init__(model=model, device=device, **kwargs)
        self.model_name = model
        self.device = device

    def query(
            self,
            user_message: Union[Message, List[Message]],
            system_prompt: str = None,
            *args,
            **kwargs,
    ) -> Union[str, List[str]]:
        """
        Format the input and call the Open-sourced LLMs.

        args:
            user_message: the user message
            system_prompt: the system prompt
        """
        if isinstance(user_message, Message):
            all_messages = []
            assert user_message.agent_name == "user"
            if "llama" in self.model_name.lower():
                if system_prompt:
                    all_messages.append(self._msg_template(SYSTEM_NAME, system_prompt))
                all_messages.append(user_message.content)
            else:
                all_messages.append([system_prompt + user_message.content if system_prompt else user_message.content])

            conversation = Conversation(
                text=all_messages[-1],
                past_user_inputs=all_messages[:-1],
                generated_responses=[],
            )
            response = self._get_response(conversation)
        else:
            if "llama" in self.model_name.lower():
                if system_prompt:
                    all_messages_batch = [
                        [self._msg_template(SYSTEM_NAME, system_prompt)] for _ in range(len(user_message))]
                else:
                    all_messages_batch = [[] for _ in range(len(user_message))]
                for i, message in enumerate(user_message):
                    assert message.agent_name == "user"
                    all_messages_batch[i].append(message.content)
            else:
                all_messages_batch = []
                for i, message in enumerate(user_message):
                    assert message.agent_name == "user"
                    all_messages_batch.append(
                        [system_prompt + message.content if system_prompt else message.content])
            conversation_batch = [Conversation(
                text=all_messages[-1],
                past_user_inputs=all_messages[:-1],
                generated_responses=[],
            ) for all_messages in all_messages_batch]

            response = self._get_response(conversation_batch)

        return response


class OpenLLMInteract(TransformersConversational):
    stateful = False
    type_name = "opensource-interact"

    def __init__(
            self,
            temperature: float = DEFAULT_TEMPERATURE,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            model: str = DEFAULT_MODEL,
            device: int = -1,
            **kwargs,
    ):
        """Instantiate the OpenLLMChat backend.
        args:
            temperature: the temperature of the sampling
            max_tokens: the maximum number of tokens to sample
            model: the model to use
            merge_other_agents_as_one_user: whether to merge messages from other agents as one user message
        """
        kwargs = kwargs if kwargs else {}
        kwargs["max_tokens"] = max_tokens
        if temperature > 0:
            kwargs["temperature"] = temperature

        super().__init__(model=model, device=device, **kwargs)
        self.model_name = model
        self.device = device

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
        user_inputs, generated_responses = [], []
        all_messages = (
            [(SYSTEM_NAME, global_prompt), (SYSTEM_NAME, role_desc)]
            if global_prompt
            else [(SYSTEM_NAME, role_desc)]
        )

        total_words = 5300 - len(role_desc.split())
        if global_prompt:
            total_words -= len(global_prompt.split())
        history_messages_len = [len(msg.content.split()) + 2 for msg in history_messages]
        if sum(history_messages_len) > total_words:
            current_total = 0
            for i in range(len(history_messages_len) - 1, -1, -1):
                current_total += history_messages_len[i]
                if current_total > total_words:
                    history_messages = history_messages[i + 1:]
                    break

        for msg in history_messages:
            all_messages.append((msg.agent_name, msg.content))
        if request_msg:
            all_messages.append((SYSTEM_NAME, request_msg.content + " Generate just one response."))

        prev_is_user = False  # Whether the previous message is from the user
        for i, message in enumerate(all_messages):
            if i == 0:
                assert (
                        message[0] == SYSTEM_NAME
                )  # The first message should be from the system

            if message[0] != agent_name:
                if not prev_is_user:
                    user_inputs.append(self._msg_template(message[0], message[1]))
                else:
                    user_inputs[-1] += "\n" + self._msg_template(message[0], message[1])
                prev_is_user = True
            else:
                if prev_is_user:
                    generated_responses.append(self._msg_template(message[0], message[1]))
                else:
                    generated_responses[-1] += "\n" + self._msg_template(message[0], message[1])
                prev_is_user = False

        assert len(user_inputs) == len(generated_responses) + 1
        past_user_inputs = user_inputs[:-1]
        new_user_input = user_inputs[-1]

        # Recreate a conversation object from the history messages
        conversation = Conversation(
            text=new_user_input,
            past_user_inputs=past_user_inputs,
            generated_responses=generated_responses,
        )

        # Get the response
        response = self._get_response(conversation)
        response = re.sub(rf"^\s*\[.*]:", "", response).strip()  # noqa: F541
        response = re.sub(
            rf"^\s*{re.escape(agent_name)}\s*:", "", response, 1
        ).strip()  # noqa: F451
        return response
