import sys
import os
import re
import json
from tqdm import tqdm
from typing import List
import torch
import argparse
import logging

args = argparse.ArgumentParser()
args.add_argument("--client_model", type=str, default="Mistral")
args.add_argument("--client_url", type=str, default=None)  # "http://0.0.0.0:50072/v1/chat/completions"
args.add_argument("--therapist_model", type=str, default="LLaMA-3")
args.add_argument("--therapist_url", type=str, default=None)  # "http://0.0.0.0:50073/v1/chat/completions"
args.add_argument("--info_extracted_model", type=str, default="claude")
# choices = [
#     "gpt-4-0125-preview", "gpt-3.5-turbo-0125", "claude-3-opus-20240229"
# ]
# /home/share/models/Meta-Llama-3-8B-Instruct
# /home/share/models/Mistral-7B-Instruct-v0.2
args.add_argument("--dataset_name", type=str, default="HighLow_transcript")
args.add_argument("--output_dir", type=str, default="simulation_conversations/test")
args.add_argument("--temperature", type=float, default=0.7)
args.add_argument("--fp32", action='store_true')
args = args.parse_args()

os.environ["ANTHROPIC_API_KEY"] = json.load(open("api_key.json", "r", encoding="utf-8"))["claude_api_key"]
os.environ["OPENAI_API_KEY"] = json.load(open("api_key.json", "r", encoding="utf-8"))["openai_api_key"]
os.environ["OPENAI_API_BASE"] = json.load(open("api_key.json", "r", encoding="utf-8"))["openai_api_base"]
sys.path.append("../")

logging.getLogger().setLevel(logging.INFO)

from chatarena.arena import Arena
from chatarena.environments.base import Environment, TimeStep
from chatarena.message import Message, MessagePool
from client_simulation import SimulatedClient
from therapist_simulation import SimulatedTherapist
from llm_utils import VLLMInteract, GPTInteract, ClaudeInteract

if args.client_model == "GPT-3.5":
    ClientAIChat = GPTInteract(max_tokens=512, model="gpt-3.5-turbo-0125", temperature=args.temperature)
elif args.client_model == "Claude-3":
    ClientAIChat = ClaudeInteract(max_tokens=512, model="claude-3-haiku-20240307", temperature=args.temperature)
else:
    ClientAIChat = VLLMInteract(
        max_tokens=512, model=args.client_model, model_url=args.client_url, temperature=args.temperature)
# else:
#     raise ValueError("--client_model is not valid.")

if args.therapist_model == "GPT-3.5":
    TherapistAIChat = GPTInteract(max_tokens=512, model="gpt-3.5-turbo-0125", temperature=args.temperature)
elif args.therapist_model == "Claude-3":
    TherapistAIChat = ClaudeInteract(max_tokens=512, model="claude-3-haiku-20240307", temperature=args.temperature)
else:
    TherapistAIChat = VLLMInteract(
        max_tokens=512, model=args.therapist_model, model_url=args.therapist_url, temperature=args.temperature)
# else:
#     raise ValueError("--therapist_model is not valid.")
SPEAKER = {"T": "Therapist", "C": "Client"}


class Counseling(Environment):
    type_name = "counseling"

    def __init__(self, start_of_conv, max_turns: int = 100):
        super().__init__(player_names=["Client", "Therapist"])
        # The "state" of the environment is maintained by the message pool
        self.message_pool = MessagePool()
        self.max_turns = max_turns
        self._terminal = False
        self._start_of_conv = start_of_conv
        self.reset()
        self.last_code = ""

    def get_next_player(self) -> str:
        return self._start_of_conv[(self.turn + 1) % 2]["role"]

    def reset(self):
        self.turn = 0
        self.message_pool.reset()
        self.turn += 1
        self.message_pool.append_message(
            Message(self._start_of_conv[0]["role"], self._start_of_conv[0]["content"], self.turn)
        )
        self.turn += 1
        self.message_pool.append_message(
            Message(self._start_of_conv[1]["role"], self._start_of_conv[1]["content"], self.turn)
        )
        observation = self.get_observation(self.get_next_player())
        self._terminal = False
        self.turn += 1
        return TimeStep(
            observation=observation,
            reward=self.get_zero_rewards(),
            terminal=self._terminal,
        )

    def get_observation(self, player_name=None) -> List[Message]:
        if player_name is None:
            return self.message_pool.get_all_messages()
        else:
            return self.message_pool.get_visible_messages(
                player_name, turn=self.turn + 1
            )

    def process_broken(self):
        self._terminal = True
        observation = self.get_observation(self.get_next_player())
        return TimeStep(
            observation=observation,
            reward=self.get_zero_rewards(),
            terminal=self._terminal,
        )

    def session_end(self):
        self._terminal = True
        return TimeStep(
            observation=self.get_observation(self.get_next_player()),
            reward=self.get_one_rewards(),
            terminal=self._terminal,
        )

    def step(self, player_name: str, action: str) -> TimeStep:
        assert (
                player_name == self.get_next_player()
        ), f"Wrong player! It is {self.get_next_player()} turn."
        visible_to = "all"
        action = self.remove_noise(action, player_name)
        if "[END OF SESSION]" in action or "<<<<<<END_OF_CONVERSATION>>>>>>" in action:
            self.session_end()
            if "<<<<<<END_OF_CONVERSATION>>>>>>" in action:
                exit()
        message = Message(
            agent_name=player_name,
            content=action,
            turn=self.turn,
            visible_to=visible_to,
        )
        logging.info(f"{player_name}: {action}")
        self.message_pool.append_message(message)
        self.turn += 1
        return TimeStep(
            observation=self.get_observation(self.get_next_player()),
            reward=self.get_zero_rewards(),
            terminal=self._terminal,
        )

    def remove_noise(self, action, player_name):
        action = re.sub(r'(\s?)\*[^*]*\*(\s?)', lambda m: ' ' if m.group(1) and m.group(2) else '', action)
        action = re.sub(r'(\s?)\(.*?\)(\s?)', lambda m: ' ' if m.group(1) and m.group(2) else '', action)
        action = action.replace("\n\n", "\n")
        action = action.replace("\n", " ")
        if "[Client]: " in action:
            action = action.split("[Client]")[0]
        if "[Therapist]: " in action:
            action = action.split("[Therapist]")[0]
        if "[Client] " in action and player_name == "Client":
            action = action.split("[Client]")[0]
        if "[Therapist] " in action and player_name == "Therapist":
            action = action.split("[Therapist]")[0]
        return action.strip()


def remove_noise(text):
    text = re.sub(r'(\s?)\[.*?\](\s?)', lambda m: ' ' if m.group(1) and m.group(2) else '', text)
    return text.strip()


if __name__ == "__main__":
    file_list = os.listdir(
        f"client_simulation/output/{args.info_extracted_model}_annotated/{args.dataset_name}/profile_formatted")
    logging.info("Simulation content found in %s" % args.info_extracted_model)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    for conversation_index in tqdm(sorted(file_list), total=len(file_list)):
        conversation_index = conversation_index.split(".")[0]
        if os.path.exists(f"{args.output_dir}/{conversation_index}.json"):
            continue
        logging.info(f"Conversation {conversation_index}")
        conversation_path = f"dataset/{args.dataset_name}/{conversation_index}.txt"
        conversation = open(conversation_path, "r", encoding="utf-8").readlines()
        conversation = [remove_noise(line) for line in conversation]
        first_utt = conversation[0]
        second_utt = conversation[1]

        client_info_file = f"client_simulation/output/{args.info_extracted_model}_annotated/{args.dataset_name}/%s/{conversation_index}.json"
        client = SimulatedClient(
            profile_path=client_info_file % ("profile_formatted"),
            personality_path=client_info_file % ("big_five_formatted"),
            symptom_path=client_info_file % ("symptoms_formatted"),
            backend=ClientAIChat,
            conversation=conversation
        )

        therapist = SimulatedTherapist(
            backend=TherapistAIChat,
            client_path=client_info_file % ("profile_formatted")
        )
        env = Counseling([
            {"content": first_utt[3:].strip(), "role": SPEAKER[first_utt[0]]},
            {"content": second_utt[3:].strip(), "role": SPEAKER[second_utt[0]]}
        ])
        arena = Arena([client, therapist], env)
        arena.run(int(len(conversation) * 1.05))
        arena.save_history(f"{args.output_dir}/{conversation_index}.json")
