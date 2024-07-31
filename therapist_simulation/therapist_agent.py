import json
import os
from typing import Union, List
import re
import sys
import argparse
import logging

sys.path.append("../")

from tenacity import RetryError
from chatarena.agent import Player, BackendConfig, IntelligenceBackend
from chatarena.message import Message
from llm_utils import SIGNAL_END_OF_CONVERSATION

logging.getLogger().setLevel(logging.INFO)

PROMPT_PATH = re.sub(r'\.py$', ".json", __file__)
SIMULATION_PROMPTS = json.load(open(PROMPT_PATH, "r", encoding="utf-8"))
SYSTEM_PROMPT = SIMULATION_PROMPTS["simulation instruction"]
ATTENTION_PROMPT = SIMULATION_PROMPTS["attention"]
INSTANCE_PROMPT = SIMULATION_PROMPTS["instance"]


class SimulatedTherapist(Player):
    def __init__(
            self,
            backend: Union[BackendConfig, IntelligenceBackend],
            global_prompt: str = None,
            conversation: List[str] = None,
            client_path: str = None,
            **kwargs
    ):
        name = "Therapist"
        client_information = {}
        if client_path:
            client_profile = json.load(open(client_path, "r", encoding="utf-8"))
            if "name" in client_profile:
                client_information["name"] = client_profile["name"]
            if "gender" in client_profile:
                client_information["gender"] = client_profile["gender"]

        if conversation and len(conversation) >= 60:
            role_desc = self.therapist_role_desc(conversation[:50], client_information)
            self.conv_index = 50
        else:
            role_desc = self.therapist_role_desc(conversation, client_information)
            self.conv_index = len(conversation) if conversation else 0
        super().__init__(name, role_desc, backend, global_prompt, **kwargs)
        self.conversation = conversation
        self.client_information = client_information

    def therapist_role_desc(self, conversation=None, client_information=None):
        if conversation:
            new_conv = []
            for line in conversation:
                if line.startswith("T: "):
                    new_conv.append("[Teacher Therapist]: " + line[3:].strip())
                else:
                    new_conv.append("[Another Client]: " + line[3:].strip())
            conv_string = "\n".join(new_conv)
            instance = INSTANCE_PROMPT.format(conversation=conv_string)
        else:
            instance = ""
        if client_information:
            client_info = "There are some basic information about the client ([Client]): the client's"
            for key, value in client_information.items():
                client_info += f" {key} is {value}, and the"
            client_info = client_info[:-9] + ".\n\n"
        else:
            client_info = ""
        role_desc = SYSTEM_PROMPT.format(
            instance=instance,
            client_info=client_info
        )
        return role_desc

    def act(self, observation: List[Message], request_msg: Message = None) -> str:
        """
                Take an action based on the observation (Generate a response), which can later be parsed to actual actions that affect the game dynamics.

                Parameters:
                    observation (List[Message]): The messages that the player has observed from the environment.

                Returns:
                    str: The action (response) of the player.
                """
        if self.conversation and self.conv_index * 0.85 < len(observation) and self.conv_index < len(self.conversation):
            self.role_desc = self.therapist_role_desc(
                self.conversation[self.conv_index - 10: self.conv_index + 50], self.client_information)
            self.conv_index = min(len(self.conversation), self.conv_index + 50)
        try:
            response = self.backend.query(
                agent_name=self.name,
                role_desc=self.role_desc,
                history_messages=observation,
                global_prompt=self.global_prompt,
                request_msg=Message(
                    agent_name="user",
                    content=ATTENTION_PROMPT if self.conversation else "Now, [Therapist] speaks, and maintain the style of your previous responses as well. Avoid either a long response or a rambling statement.",
                    turn=-1
                ),
            )
        except RetryError as e:
            err_msg = f"Agent {self.name} failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
            logging.warning(err_msg)
            response = SIGNAL_END_OF_CONVERSATION + err_msg

        return response


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_name",
        type=str, default="gpt-3.5-turbo-0125",
        choices=[
            "gpt-4-0125-preview",
            "gpt-3.5-turbo-0125",
            "claude-2.1",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229"
        ]
    )
    args.add_argument("--dataset_name", type=str, default="")
    args.add_argument("--conversation_index", type=str, default="")
    args.add_argument("--output_path", type=str, default="")
    args = args.parse_args()

    os.environ["ANTHROPIC_API_KEY"] = json.load(open("api_key.json", "r", encoding="utf-8"))["claude_api_key2"]
    os.environ["OPENAI_API_KEY"] = json.load(open("api_key.json", "r", encoding="utf-8"))["openai_api_key"]
    os.environ["OPENAI_API_BASE"] = json.load(open("api_key.json", "r", encoding="utf-8"))["openai_api_base"]

    sys.path.append("../")
