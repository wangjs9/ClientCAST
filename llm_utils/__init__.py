import os
import uuid
from typing import List, Union
from tenacity import RetryError

ROOT_DIR = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + os.path.sep
)
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"

from chatarena.message import Message
from .api_utils import ClaudeChat, ClaudeInteract, GPTChat, GPTInteract
from .open_utils import OpenLLMChat, OpenLLMInteract
from .human_utils import HumanInteract
from .vllm_utils import VLLMInteract


def llm_response(
        llm_agent: Union[GPTChat, ClaudeChat, OpenLLMChat],
        query_message: Union[Message, List[Message]],
        system_prompt: str = None,
        model_name=None
) -> str:
    try:
        response = llm_agent.query(
            user_message=query_message,
            system_prompt=system_prompt
        )
    except RetryError as e:
        if model_name:
            err_msg = f"Agent {model_name} failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
        else:
            err_msg = f"Agent failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
        response = SIGNAL_END_OF_CONVERSATION + err_msg
    return response


def llm_chat(
        agent_name: str,
        llm_agent: Union[GPTInteract, ClaudeInteract, OpenLLMInteract],
        history_messages: List[Message],
        request_msg: str = None,
        model_name=None
) -> str:
    all_messages = []
    for msg in history_messages:
        if msg.agent_name == agent_name:
            msg.agent_name = "system"
            all_messages.append(("system", msg.content))
        else:
            all_messages.append((msg.agent_name, msg.content))

    if request_msg:
        last_msg = all_messages.pop(-1)
        all_messages.append((last_msg[0], last_msg[1] + "\n" + request_msg))

    try:
        response = llm_agent.query(
            user_messages=all_messages
        )
    except RetryError as e:
        if model_name:
            err_msg = f"Agent {model_name} failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
        else:
            err_msg = f"Agent failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
        response = SIGNAL_END_OF_CONVERSATION + err_msg

    return response
