import os
import re
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME as SYSTEM
from ..message import Message
from .base import IntelligenceBackend, register_backend

try:
    import anthropic
except ImportError:
    is_anthropic_available = False
    # logging.warning("anthropic package is not installed")
else:
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    if anthropic_api_key is None:
        # logging.warning("Anthropic API key is not set. Please set the environment variable ANTHROPIC_API_KEY")
        is_anthropic_available = False
    else:
        is_anthropic_available = True

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "claude-v1"


@register_backend
class Claude(IntelligenceBackend):
    """Interface to the Claude offered by Anthropic."""

    stateful = False
    type_name = "claude"

    def __init__(
            self,
            temperature: float = DEFAULT_TEMPERATURE,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            model: str = DEFAULT_MODEL,
            **kwargs
    ):
        assert (
            is_anthropic_available
        ), "anthropic package is not installed or the API key is not set"
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            **kwargs
        )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model

    # @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=60, exp_base=2, min=30, max=120))
    def _get_response(self, messages):
        if messages[0]["role"] == "system":
            system = messages.pop(0)["content"]
            completion = client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            completion = client.messages.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        response = completion.content[0].text
        response = response.strip()
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
        """
        Format the input and call the Claude API.

        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request from the system to guide the agent's next response
        """
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt = f"You are a helpful assistant.\n{global_prompt.strip()}\n{BASE_PROMPT}\n\nYour name is {agent_name}.\n\nYour role:{role_desc}"
        else:
            system_prompt = f"You are a helpful assistant. Your name is {agent_name}.\n\nYour role:{role_desc}\n\n{BASE_PROMPT}"

        all_messages = []
        for message in history_messages:
            all_messages.append((message.agent_name, message.content))
        if request_msg:
            all_messages.append((SYSTEM, request_msg.content))

        prompt = ""
        prev_is_human = False  # Whether the previous message is from human (in anthropic, the human is the user)
        for i, message in enumerate(all_messages):
            if i == 0:
                assert (
                        message[0] == SYSTEM
                )  # The first message should be from the system

            if message[0] == agent_name:
                if prev_is_human:
                    prompt = f"{prompt}{anthropic.AI_PROMPT} {message[1]}"
                else:
                    prompt = f"{prompt}\n\n{message[1]}"
                prev_is_human = False
            else:
                if prev_is_human:
                    prompt = f"{prompt}\n\n[{message[0]}]: {message[1]}"
                else:
                    prompt = f"{prompt}{anthropic.HUMAN_PROMPT}\n[{message[0]}]: {message[1]}"
                prev_is_human = True
        assert prev_is_human  # The last message should be from the human
        # Add the AI prompt for Claude to generate the response
        prompt = f"{prompt}{anthropic.AI_PROMPT}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self._get_response(messages)

        # Remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[{agent_name}]:?", "", response).strip()

        return response
