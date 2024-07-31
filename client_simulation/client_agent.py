import json
import os
import re
import functools
from typing import Union, List
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
APPEARANCE_PROMPT = SIMULATION_PROMPTS["appearance"]
SYMPTOM_PROMPT = SIMULATION_PROMPTS["symptoms"]
OTHERS_PROMPT = SIMULATION_PROMPTS["others"]
ATTENTION_PROMPT = SIMULATION_PROMPTS["attention"]
INSTANCE_PROMPT = SIMULATION_PROMPTS["instance"]


def only_run_once(func):
    @functools.wraps(func)
    def wrapper_only_run_once(*args, **kwargs):
        if wrapper_only_run_once.first_call == 0:
            return wrapper_only_run_once.result
        else:
            wrapper_only_run_once.first_call = 0
            wrapper_only_run_once.result = func(*args, **kwargs)
            return wrapper_only_run_once.result

    wrapper_only_run_once.first_call = 1
    wrapper_only_run_once.result = 0
    return wrapper_only_run_once


class SimulatedClient(Player):
    def __init__(
            self,
            profile_path: str,
            personality_path: str,
            symptom_path: str,
            backend: Union[BackendConfig, IntelligenceBackend],
            conversation: List[str] = None,
            global_prompt: str = None,
            **kwargs
    ):
        """
        profile_path: the path to the client's profile, including the basic information of the client and the symptoms
        personality_path: the path to the client's personality traits
        model_name: the model name
        """
        name = "Client"
        profile = json.load(open(profile_path, "r", encoding="utf-8"))
        personality = json.load(open(personality_path, "r", encoding="utf-8"))
        symptoms = json.load(open(symptom_path, "r", encoding="utf-8"))

        if len(conversation) >= 60:
            role_desc = self.client_role_desc(profile, personality, symptoms, conversation[:50])
            self.conv_index = 50
        else:
            role_desc = self.client_role_desc(profile, personality, symptoms, conversation)
            self.conv_index = len(conversation)
        super().__init__(name, role_desc, backend, global_prompt, **kwargs)
        self.conversation = conversation
        self.profile = profile
        self.personality = personality
        self.symptoms = symptoms

    def client_role_desc(
            self,
            profile: dict,
            personality: dict,
            symptoms: dict,
            conversation: List[str] = None
    ):
        name = profile.get("name", "the client")
        case_synopsis = f"You are {name}"
        if 'gender' in profile:
            case_synopsis += f", a {profile['gender']}"
        if 'age' in profile:
            case_synopsis += f", aged {profile['age']}"
        if 'occupation' in profile:
            case_synopsis += f", and work as a {profile['occupation']}"
        case_synopsis += f". You are currently experiencing the problem of {profile['problem']} {profile['emotion']}"

        other_instruction = ""

        appearance = ""
        if "feeling expression" in profile:
            appearance += f"\n- {profile['feeling expression']}"
        else:
            appearance += f"\n- You can be unwilling to express your feelings."
        if "emotional fluctuation" in profile:
            appearance += f"\n- {profile['emotional fluctuation']}"
        else:
            appearance += f"\n- You can have emotional fluctuations during the conversation."
        if "resistance" in profile:
            appearance += f"\n- {profile['resistance']}"
        else:
            appearance += f"\n- You can have a resistance towards the therapist, and do not want to reveal some feelings easily."
        if personality:
            appearance += "\n- " + "\n- ".join([trait for trait in personality.values()])
        if appearance:
            appearance = APPEARANCE_PROMPT.format(appearance=appearance)
            other_instruction += appearance

        if symptoms:
            client_symptoms = ""
            for line in symptoms:
                client_symptoms += f"- \"{line['symptom']}\" with the severity of \"{line['severity']}\". The detailed appearance can be: {line['explanation']}\n"
            client_symptoms = SYMPTOM_PROMPT.format(symptoms="\n" + client_symptoms)
            other_instruction += client_symptoms

        if conversation:
            new_conv = []
            for line in conversation:
                if line.startswith("T: "):
                    new_conv.append("[Another Therapist]: " + line[3:].strip())
                else:
                    new_conv.append("[Original Client]: " + line[3:].strip())
            conv_string = "\n".join(new_conv)
            instance = INSTANCE_PROMPT.format(conversation=conv_string)
        else:
            instance = ""

        role_desc = SYSTEM_PROMPT.format(
            case_synopsis=case_synopsis,
            reasons4visit=profile["reasons"],
            other_instruction=other_instruction,
            instance=instance
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
        if self.conv_index * 0.85 < len(observation) and self.conv_index < len(self.conversation):
            self.role_desc = self.client_role_desc(
                self.profile, self.personality, self.symptoms,
                self.conversation[self.conv_index - 10:self.conv_index + 50])
            self.conv_index = min(len(self.conversation), self.conv_index + 50)

        try:
            response = self.backend.query(
                agent_name=self.name,
                role_desc=self.role_desc,
                history_messages=observation,
                global_prompt=self.global_prompt,
                request_msg=Message(agent_name="user", content=ATTENTION_PROMPT, turn=-1),
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

    profile_path = "./output/AnnoMI_transcript/profile_formatted/high_021.json"
    personality_path = "./output/AnnoMI_transcript/big_five_formatted/high_021.json"
    symptom_path = "./output/AnnoMI_transcript/symptoms_formatted/high_021.json"
    client = SimulatedClient(profile_path, personality_path, symptom_path)
