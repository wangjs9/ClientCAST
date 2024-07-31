"""
This script is used to evaluate the performance of the client agent from the aspect of clients.
"""
import os
import sys
import json
import uuid
import logging
from tqdm import tqdm
import torch
import argparse

args = argparse.ArgumentParser()
args.add_argument("--model_name", type=str,
                  default="/models/Meta-Llama-3-70B-Instruct")
# gpt-3.5-turbo-0125
args.add_argument("--questionnaire_name", type=str, choices=["CECS", "HAQ-II", "SEQ", "SRS", "WAI-SR", "all"])
args.add_argument("--conv_path", type=str, default="../dataset/AnnoMI_transcript")
args.add_argument("--device", type=str, default="4")
args.add_argument("--output_path", type=str, default="")
args.add_argument(
    "--profile_dir", type=str,
    default="../client_simulation/output/gpt_annotated/AnnoMI_transcript/profile_formatted/")
args.add_argument(
    "--personality_dir", type=str,
    default="../client_simulation/output/gpt_annotated/AnnoMI_transcript/big_five_formatted/")
args = args.parse_args()

os.environ["ANTHROPIC_API_KEY"] = json.load(open("../api_key.json", "r", encoding="utf-8"))["claude_api_key"]
os.environ["OPENAI_API_KEY"] = json.load(open("../api_key.json", "r", encoding="utf-8"))["openai_api_key"]
os.environ["OPENAI_API_BASE"] = json.load(open("../api_key.json", "r", encoding="utf-8"))["openai_api_base"]
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

sys.path.append("../")

from chatarena.message import Message
from llm_utils import GPTChat, ClaudeChat, OpenLLMChat, llm_response
from client_simulation.utils.format_utils import load_corpus, conv_list2str

if "gpt" in args.model_name:
    Agent = GPTChat
elif "claude" in args.model_name:
    Agent = ClaudeChat
elif os.path.exists(args.model_name):
    Agent = OpenLLMChat
else:
    raise ValueError("Model not supported")

logging.getLogger().setLevel(logging.INFO)

SPEAKER = {"T": "Therapist", "C": "Client"}
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>"


def conduct_questionnaire(llm_agent, system_prompt, queries, scale, reference_res=None):
    # set the model
    if isinstance(queries, dict):
        query_prefixs = list(queries.keys())
        for prefix in query_prefixs:
            for query in queries[prefix]:
                query.update({"prefix": prefix})
        queries = sum(list(queries.values()), [])
    assert isinstance(queries, list)
    questionnaire_res = []
    if os.path.exists(args.model_name):
        batch_size = 16
        for i in range(0, len(queries), batch_size):
            query_batch = queries[i:i + batch_size]
            query_content_batch = [query["content"] for query in query_batch]
            query_scale_batch = [query.get("scale", scale) for query in query_batch]
            prefix_batch = [query.get("prefix", "") for query in query_batch]
            prefix_batch = [prefix.lower().capitalize() if prefix else "" for prefix in prefix_batch]
            query_content_batch = [
                f"{prefix}, {query_content[0].lower() + query_content[1:]}" if prefix else query_content for
                prefix, query_content in zip(prefix_batch, query_content_batch)]
            query_prompt_batch = []
            for query_content, query_scale in zip(query_content_batch, query_scale_batch):
                if query_scale and isinstance(query_scale, dict):
                    query_prompt_batch.append(
                        f"{query_content} {', '.join([f'{k}: {v}' for k, v in query_scale.items()])[:-1]}.")
                elif isinstance(query_scale, str):
                    query_prompt_batch.append(f"{query_content} {query_scale}")
                else:
                    query_prompt_batch.append(query_content)
            query_message_batch = [Message(
                agent_name="user",
                content=query_prompt + "\nAttention: Start your response with the rating.",
                turn=-1) for query_prompt in query_prompt_batch]
            response_batch = llm_response(llm_agent, query_message_batch, system_prompt)
            # print(response_batch)
            assert len(query_batch) == len(response_batch), response_batch
            for query, response in zip(query_batch, response_batch):
                query.update({"response": response})
                questionnaire_res.append(query)
    else:
        for idx, query in enumerate(queries):
            if reference_res and SIGNAL_END_OF_CONVERSATION not in reference_res[idx]["response"]:
                questionnaire_res.append(reference_res[idx])
            else:
                query_content = query["content"]
                query_scale = query.get("scale", scale)
                prefix = query.get("prefix", "")
                if prefix:
                    prefix = prefix.lower().capitalize()
                    query_content = f"{prefix}, {query_content[0].lower() + query_content[1:]}"
                if query_scale and isinstance(query_scale, dict):
                    query_prompt = f"{query_content} {', '.join([f'{k}: {v}' for k, v in query_scale.items()])[:-1]}."
                elif isinstance(query_scale, str):
                    query_prompt = f"{query_content} {query_scale}"
                else:
                    query_prompt = query_content
                query_message = Message(
                    agent_name="user",
                    content=query_prompt + "\nAttention: Start your response with the rating.",
                    turn=-1
                )
                response = llm_response(llm_agent, query_message, system_prompt)
                query.update({"response": response})
                questionnaire_res.append(query)

    return questionnaire_res


def load_character(profile_json: dict, personality: dict) -> str:
    """
    Load the profile/personalities from the profile/personality json (a dict)
    """
    character_string = ""
    name = profile_json.get("name", None)
    if name:
        character_string = f"Your name is {name}"
    gender = profile_json.get("gender", None)
    if gender and character_string:
        character_string += f", a {gender}"
    elif gender:
        character_string = f"You are a {gender}"
    age = profile_json.get("age", None)
    if age and character_string:
        character_string += f", aged {age}"
    elif age:
        character_string = f"You are aged {age}"
    occupation = profile_json.get("occupation", None)
    if occupation and character_string:
        character_string += f", and work as a {occupation}"
    elif occupation:
        character_string = f"You are working as a {occupation}"
    reasons = profile_json["reasons"]
    if character_string:
        character_string += f". Reasons For Your Visiting: {reasons} "
    else:
        character_string = f"Reasons For Your Visiting: {reasons} "

    if personality:
        character_string += "You have the following personality traits: "
        for trait, value in personality.items():
            character_string += f"{value}, "
        character_string = character_string[:-2] + "."

    return character_string


def complete_one(questionnaire_name, llm_agent, model_name, output_path=""):
    # load the questionnaire content
    logging.info(f"Conducting questionnaire {questionnaire_name}...")
    questionnaire = json.load(open(f"questionnaires/{questionnaire_name}.json", "r"))
    system_prompt_list = questionnaire["system_prompt"]
    queries = questionnaire["queries"]
    scale = questionnaire.get("scale", None)

    # load conversation
    logging.info(f"Loading the conversation corpus from {args.conv_path}...")
    conv_path_list = load_corpus(args.conv_path)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
    else:
        dataset = args.conv_path.split("/")[-1]
        output_path = f"questionnaire_results/{model_name}_completed/{dataset}/{questionnaire_name}/"
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
    logging.info(f"Saving the results to {output_path}...")

    # iterate over the conversations
    logging.info(f"Loading the profile/personality from {args.profile_dir} and {args.personality_dir}...")

    for conv_idx, conv_path in tqdm(enumerate(conv_path_list), total=len(conv_path_list)):
        save_path = conv_path.split("/")[-1].replace(".txt", ".json")
        reference_res = None
        if os.path.exists(os.path.join(output_path, save_path)):
            finished = True
            reference_res = json.load(open(os.path.join(output_path, save_path), "r"))
            for item in reference_res:
                if SIGNAL_END_OF_CONVERSATION in item["response"]:
                    finished = False
                    break
            if finished:
                continue

        # load the conversation
        conv = open(conv_path, "r").readlines()
        conv_string = conv_list2str(conv)
        if len(conv_string.split()) > 4200:
            conv_string = " ".join(conv_string.split(" ")[:4200]).replace("\n ", "\n")
        # load the profile and personalities
        profile = json.load(open(os.path.join(args.profile_dir, save_path), "r"))
        personality = json.load(open(os.path.join(args.personality_dir, save_path), "r"))
        client_character = load_character(profile, personality)
        system_prompt = system_prompt_list["content"].format(
            character=client_character,
            conversation=conv_string,
            questionnaire_desc=system_prompt_list["questionnaire description"]
        )
        # conduct the questionnaire
        questionnaire_res = conduct_questionnaire(llm_agent, system_prompt, queries, scale, reference_res)
        # save the result
        json.dump(questionnaire_res, open(os.path.join(output_path, save_path), "w"), indent=2)


def main(args: argparse.Namespace):
    # set model
    torch.cuda.empty_cache()
    llm_agent = Agent(temperature=0, model=args.model_name, max_tokens=50)
    logging.info(f"Model {args.model_name} loaded successfully.")

    # prepare the output path
    model_name = "claude"
    if "gpt" in args.model_name:
        model_name = "gpt"
    elif "llama-3-8b" in args.model_name.lower():
        model_name = "llama"
    elif "llama-3-70b" in args.model_name.lower():
        model_name = "llama-70b"
    elif "mistral" in args.model_name.lower():
        model_name = "mistral"
    elif "Mixtral" in args.model_name:
        model_name = "Mixtral"

    if args.questionnaire_name == "all":
        for questionnaire_name in ["CECS", "HAQ-II", "SEQ", "SRS", "WAI-SR"]:
            output_path = f"questionnaire_results/{model_name}_completed/{args.output_path}/{questionnaire_name}/"
            complete_one(questionnaire_name, llm_agent, model_name, output_path)
    else:
        complete_one(args.questionnaire_name, llm_agent, model_name, args.output_path)


if __name__ == "__main__":
    main(args)
