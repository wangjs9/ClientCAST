"""
This is a file to utilize the API models, including OpenAI's ChatGPT and GPT-4, and Anthropics' LLM.
"""
from typing import List, Union
import uuid
import re

try:
    import anthropic
except ImportError:
    is_anthropic_available = False

from chatarena.message import Message
from chatarena.backends import OpenAIChat, Claude

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256

SYSTEM_NAME = "System"
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class ClaudeChat(Claude):

    def query(
            self,
            user_message: Message,
            system_prompt: str = None,
            *args,
            **kwargs,
    ) -> str:
        """
        Format the input and call the Claude API.
        """
        all_messages = []
        if system_prompt:
            all_messages = [{"role": "system", "content": system_prompt}]
        assert user_message.agent_name == "user"
        all_messages.append({"role": user_message.agent_name, "content": user_message.content})
        response = self._get_response(all_messages)
        return response


class ClaudeInteract(Claude):

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
        """
        Format the input and call the ChatGPT/GPT-4 API.

        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request from the system to guide the agent's next response
        """
        # Merge the role description and the global prompt as the system prompt for the agent
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt = f"{global_prompt.strip()}\n\n{role_desc}"
        else:
            system_prompt = role_desc
        all_messages = [(SYSTEM_NAME, system_prompt)]

        total_words = 6000 - len(system_prompt.split())
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
                messages.append({"role": "system", "content": msg[1]})
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


class GPTChat(OpenAIChat):

    def query(
            self,
            user_message: Message,
            system_prompt: str = None,
            *args,
            **kwargs,
    ) -> str:
        """
        Format the input and call the ChatGPT/GPT-4 API

        args:
        user_messages: the history of the conversation
        request_msg: the request from the system to guide the agent's next response
        """
        all_messages = []
        if system_prompt:
            all_messages = [{"role": "system", "content": system_prompt}]
        assert user_message.agent_name == "user"
        all_messages.append({"role": user_message.agent_name, "content": user_message.content})
        response = self._get_response(all_messages)
        return response


class GPTInteract(OpenAIChat):
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
        """
        Format the input and call the ChatGPT/GPT-4 API.

        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request from the system to guide the agent's next response
        """
        # Merge the role description and the global prompt as the system prompt for the agent
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt = f"{global_prompt.strip()}\n\n{role_desc}"
        else:
            system_prompt = role_desc

        total_words = 6000 - len(system_prompt.split())
        history_messages_len = [len(msg.content.split()) + 2 for msg in history_messages]
        if sum(history_messages_len) > total_words:
            current_total = 0
            for i in range(len(history_messages_len) - 1, -1, -1):
                current_total += history_messages_len[i]
                if current_total > total_words:
                    history_messages = history_messages[i + 1:]
                    break

        all_messages = [(SYSTEM_NAME, system_prompt)]
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
        for i, msg in enumerate(all_messages):
            if i == 0:
                assert (
                        msg[0] == SYSTEM_NAME
                )  # The first message should be from the system
                messages.append({"role": "system", "content": msg[1]})
            else:
                if msg[0] == agent_name:
                    messages.append({"role": "assistant", "content": f"[{msg[0]}]: {msg[1]}"})
                else:
                    if msg[0] == SYSTEM_NAME:
                        if messages[-1]["role"] == "user":
                            messages[-1]["content"] = f"{messages[-1]['content']}\n\n{msg[1]}"
                    elif messages[-1]["role"] == "user":  # last message is from user
                        messages[-1]["content"] = f"{messages[-1]['content']}\n[{msg[0]}]: {msg[1]}"
                    elif messages[-1]["role"] == "system" or messages[-1]["role"] == "assistant":
                        messages.append({"role": "user", "content": f"[{msg[0]}]: {msg[1]}"})
                    else:
                        raise ValueError(f"Invalid role: {messages[-1]['role']}")
        # messages[1]["content"] = "The session starts:\n" + messages[1]["content"]
        response = self._get_response(messages)

        # Remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[.*]:", "", response).strip()  # noqa: F541
        response = re.sub(
            rf"^\s*{re.escape(agent_name)}\s*:", "", response, 1
        ).strip()  # noqa: F451

        return response
