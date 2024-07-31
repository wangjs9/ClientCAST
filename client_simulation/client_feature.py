import os
import re
import sys
import json

import logging
from collections import defaultdict

import torch
from tqdm import tqdm
import argparse

args = argparse.ArgumentParser()
args.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
# choices=["gpt-4-0125-preview", "gpt-3.5-turbo-0125",  "claude-3-haiku-20240307"]
args.add_argument("--extracted_info", type=str, choices=["all", "profile", "big_five", "symptoms"])
args.add_argument("--conv_path", type=str, required=True)
# choices=["../dataset/AnnoMI_transcript", "../dataset/HighLow_transcript", "../dataset/HOPE"])
args.add_argument("--output_path", type=str, default="")
args.add_argument("--device", type=str, default="4")
args = args.parse_args()

os.environ["ANTHROPIC_API_KEY"] = json.load(open("../api_key.json", "r", encoding="utf-8"))["claude_api_key"]
os.environ["OPENAI_API_KEY"] = json.load(open("../api_key.json", "r", encoding="utf-8"))["openai_api_key"]
os.environ["OPENAI_API_BASE"] = json.load(open("../api_key.json", "r", encoding="utf-8"))["openai_api_base"]
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

sys.path.append("../")

from chatarena.message import Message
from llm_utils import GPTChat, ClaudeChat, OpenLLMChat, llm_response
from client_simulation.utils.format_utils import conv_list2str, load_corpus

if "gpt" in args.model_name:
    Agent = GPTChat
elif "claude" in args.model_name:
    Agent = ClaudeChat
elif os.path.exists(args.model_name):
    Agent = OpenLLMChat
else:
    raise ValueError("Model not supported")

logging.getLogger().setLevel(logging.INFO)


def symptom_identification(args: argparse.Namespace, llm_agent: Agent):
    """
    In this function, we would like to summarize the client's symptoms, including PHQ-9, GAD-7, and OQ-45.

    """
    # load the conversation
    conv_path_list = load_corpus(args.conv_path)
    # load the extracted_info content
    extracted_info = json.load(open(f"simulated_content/symptoms.json", "r", encoding="utf-8"))
    system_prompt = extracted_info.get("system_prompt", "")
    # system_prompt_BDI = extracted_info.get("system_prompt_BDI", "")

    PHQ9 = extracted_info["PHQ-9"]
    PHQ9_queries = PHQ9["queries"]
    PHQ9_severity = ", ".join([f"{level}: {content}" for level, content in PHQ9["severity"].items()])

    GAD7 = extracted_info["GAD-7"]
    GAD7_queries = GAD7["queries"]
    GAD7_severity = ", ".join([f"{level}: {content}" for level, content in GAD7["severity"].items()])

    OQ45 = extracted_info["OQ-45"]
    OQ45_queries = OQ45["queries"]
    OQ45_severity_pos = ", ".join([f"{level}: {content}" for level, content in OQ45["positive severity"].items()])
    OQ45_severity_neg = ", ".join([f"{level}: {content}" for level, content in OQ45["negative severity"].items()])
    OQ45_pos_index = OQ45["positive queries"]

    # BDI = extracted_info["BDI"]

    # prepare the output path
    model_name = "claude"
    if "gpt" in args.model_name:
        model_name = "gpt"
    elif "llama" in args.model_name.lower():
        model_name = "llama"
    elif "mistral" in args.model_name.lower():
        model_name = "mistral"
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
    else:
        dataset = args.conv_path.split("/")[-1]
        args.output_path = f"output/{model_name}_annotated/{dataset}/symptoms/"
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)

    # iterate over the conversations
    if os.path.exists(args.model_name):
        batch_size = 8
        for i in tqdm(range(0, len(conv_path_list), batch_size),
                      total=(len(conv_path_list) + batch_size - 1) // batch_size):
            conv_path_batch = conv_path_list[i:i + batch_size]
            save_path_batch = [conv_path.split("/")[-1].replace(".txt", ".json") for conv_path in conv_path_batch]
            for path_idx, (save_path, conv_path) in enumerate(zip(save_path_batch, conv_path_batch)):
                if os.path.exists(f"{args.output_path}/{save_path}"):
                    conv_path_batch[path_idx] = ""
                    save_path_batch[path_idx] = ""
            conv_path_batch = [conv_path for conv_path in conv_path_batch if conv_path]
            save_path_batch = [save_path for save_path in save_path_batch if save_path]
            assert len(conv_path_batch) == len(save_path_batch)
            if len(save_path_batch) == 0:
                continue
            answer_json_batch = [defaultdict(dict) for _ in range(len(save_path_batch))]
            conv_batch = [open(conv_path, "r").readlines() for conv_path in conv_path_batch]
            conv_batch = [conv[(len(conv) - 240) // 3: (len(conv) - 240) // 3 + 240] if len(conv) > 240 else conv for
                          conv in conv_batch]
            conv_string_batch = [conv_list2str(conv) for conv in conv_batch]

            for query_idx, query in PHQ9_queries.items():
                content_batch = [system_prompt.format(conversation=conv_string, query=query, severity=PHQ9_severity) for
                                 conv_string in conv_string_batch]
                query_message_batch = [Message(agent_name="user", content=content, turn=1) for content in content_batch]
                response_batch = llm_response(llm_agent, query_message_batch)
                if isinstance(response_batch, str):
                    response_batch = [response_batch]
                assert len(response_batch) == len(save_path_batch)
                for idx, response in enumerate(response_batch):
                    answer_json_batch[idx]["PHQ-9"][query_idx] = response

            for query_idx, query in GAD7_queries.items():
                content_batch = [system_prompt.format(conversation=conv_string, query=query, severity=GAD7_severity) for
                                 conv_string in conv_string_batch]
                query_message_batch = [Message(agent_name="user", content=content, turn=1) for content in content_batch]
                response_batch = llm_response(llm_agent, query_message_batch)
                if isinstance(response_batch, str):
                    response_batch = [response_batch]
                assert len(response_batch) == len(save_path_batch)
                for idx, response in enumerate(response_batch):
                    answer_json_batch[idx]["GAD-7"][query_idx] = response

            for query_idx, query in OQ45_queries.items():
                content_batch = [system_prompt.format(conversation=conv_string, query=query,
                                                      severity=OQ45_severity_pos if query_idx in OQ45_pos_index else OQ45_severity_neg)
                                 for conv_string in conv_string_batch]
                query_message_batch = [Message(agent_name="user", content=content, turn=1) for content in content_batch]
                response_batch = llm_response(llm_agent, query_message_batch)
                if isinstance(response_batch, str):
                    response_batch = [response_batch]
                assert len(response_batch) == len(save_path_batch)
                for idx, response in enumerate(response_batch):
                    answer_json_batch[idx]["OQ-45"][query_idx] = response

            for idx, save_path in enumerate(save_path_batch):
                assert answer_json_batch[idx] != {}
                json.dump(answer_json_batch[idx], open(f"{args.output_path}/{save_path}", "w"), indent=2)

    else:
        for conv_idx, conv_path in tqdm(enumerate(conv_path_list), total=len(conv_path_list)):
            save_path = conv_path.split("/")[-1].replace(".txt", ".json")
            if os.path.exists(f"{args.output_path}/{save_path}"):
                answer_json = json.load(open(f"{args.output_path}/{save_path}", "r"))
            else:
                answer_json = defaultdict(dict)
            conv = open(conv_path, "r").readlines()
            conv_string = conv_list2str(conv)
            if len(conv_string.split(" ")) > 2800:
                conv_string = " ".join(conv_string.split(" ")[:2800]).replace("\n ", "\n")

            for query_idx, query in PHQ9_queries.items():
                response = answer_json.get("PHQ-9", {}).get(query_idx, None)
                if response is None or response == "" or "<<<<<<END_OF_CONVERSATION>>>>>>" in response:
                    content = system_prompt.format(conversation=conv_string, query=query, severity=PHQ9_severity)
                    query_message = Message(agent_name="user", content=content, turn=1)
                    # obtain answer
                    response = llm_response(llm_agent, query_message, model_name=args.model_name)
                    answer_json["PHQ-9"][query_idx] = response

            for query_idx, query in GAD7_queries.items():
                response = answer_json.get("GAD-7", {}).get(query_idx, None)
                if response is None or response == "" or "<<<<<<END_OF_CONVERSATION>>>>>>" in response:
                    content = system_prompt.format(conversation=conv_string, query=query, severity=GAD7_severity)
                    query_message = Message(agent_name="user", content=content, turn=1)
                    # obtain answer
                    response = llm_response(llm_agent, query_message, model_name=args.model_name)
                    answer_json["GAD-7"][query_idx] = response

            for query_idx, query in OQ45_queries.items():
                response = answer_json.get("OQ-45", {}).get(query_idx, None)
                if response is None or response == "" or response == "" or "<<<<<<END_OF_CONVERSATION>>>>>>" in response:
                    if query_idx in OQ45_pos_index:
                        content = system_prompt.format(
                            conversation=conv_string, query=query, severity=OQ45_severity_pos)
                    else:
                        content = system_prompt.format(
                            conversation=conv_string, query=query, severity=OQ45_severity_neg)
                    query_message = Message(agent_name="user", content=content, turn=1)
                    # obtain answer
                    response = llm_response(llm_agent, query_message, model_name=args.model_name)
                    answer_json["OQ-45"][query_idx] = response

            # save the answer
            json.dump(answer_json, open(f"{args.output_path}/{save_path}", "w"), indent=2)


def big_five_personality_traits(args: argparse.Namespace, llm_agent: Agent):
    """
    In this function, we would like to summarize the client's big five personality traits.
    """
    # load the conversation
    conv_path_list = load_corpus(args.conv_path)

    # load the extracted_info content
    extracted_info = json.load(open(f"simulated_content/big_five.json", "r", encoding="utf-8"))
    system_prompt = extracted_info.get("system_prompt", "")
    traits = extracted_info["traits"]

    # prepare the output path
    model_name = "claude"
    if "gpt" in args.model_name:
        model_name = "gpt"
    elif "llama" in args.model_name.lower():
        model_name = "llama"
    elif "mistral" in args.model_name.lower():
        model_name = "mistral"
    elif "mixtral" in args.model_name.lower():
        model_name = "Mixtral"
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
    else:
        dataset = args.conv_path.split("/")[-1]
        args.output_path = f"output/{model_name}_annotated/{dataset}/big_five/"
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)

    # iterate over the conversations
    if os.path.exists(args.model_name):
        batch_size = 8
        for i in tqdm(range(0, len(conv_path_list), batch_size),
                      total=(len(conv_path_list) + batch_size - 1) // batch_size):
            conv_path_batch = conv_path_list[i:i + batch_size]
            save_path_batch = [conv_path.split("/")[-1].replace(".txt", ".json") for conv_path in conv_path_batch]
            for path_idx, (save_path, conv_path) in enumerate(zip(save_path_batch, conv_path_batch)):
                if os.path.exists(f"{args.output_path}/{save_path}"):
                    conv_path_batch[path_idx] = ""
                    save_path_batch[path_idx] = ""
            conv_path_batch = [conv_path for conv_path in conv_path_batch if conv_path]
            save_path_batch = [save_path for save_path in save_path_batch if save_path]
            assert len(conv_path_batch) == len(save_path_batch)
            if len(save_path_batch) == 0:
                continue
            conv_batch = [open(conv_path, "r").readlines() for conv_path in conv_path_batch]
            conv_batch = [conv[(len(conv) - 240) // 3: (len(conv) - 240) // 3 + 240] if len(conv) > 240 else conv for
                          conv in conv_batch]
            conv_string_batch = [conv_list2str(conv) for conv in conv_batch]

            # expected result list
            answer_json_batch = [{} for _ in range(len(save_path_batch))]
            for trait_name, explanation in traits.items():
                content_batch = [system_prompt.format(trait_name=trait_name, explanation=explanation,
                                                      conversation=conv_string) for conv_string in conv_string_batch]
                query_message_batch = [Message(agent_name="user", content=content, turn=1) for content in content_batch]
                response_batch = llm_response(llm_agent, query_message_batch)
                if isinstance(response_batch, str):
                    response_batch = [response_batch]
                assert len(response_batch) == len(save_path_batch)
                for idx, response in enumerate(response_batch):
                    answer_json_batch[idx][trait_name] = response
            for idx, save_path in enumerate(save_path_batch):
                assert answer_json_batch[idx] != {}
                json.dump(answer_json_batch[idx], open(f"{args.output_path}/{save_path}", "w"), indent=2)

    else:
        for conv_idx, conv_path in tqdm(enumerate(conv_path_list), total=len(conv_path_list)):
            save_path = conv_path.split("/")[-1].replace(".txt", ".json")
            if os.path.exists(f"{args.output_path}/{save_path}"):
                continue
            conv = open(conv_path, "r").readlines()
            conv_string = conv_list2str(conv)
            if len(conv_string.split(" ")) > 2800:
                conv_string = " ".join(conv_string.split(" ")[:2800]).replace("\n ", "\n")

            # expected result list
            answer_json = {}
            for trait_name, explanation in traits.items():
                content = system_prompt.format(trait_name=trait_name, explanation=explanation, conversation=conv_string)
                query_message = Message(agent_name="user", content=content, turn=1)
                # obtain answer
                response = llm_response(llm_agent, query_message, model_name=args.model_name)
                answer_json[trait_name] = response

            # save the answer
            json.dump(answer_json, open(f"{args.output_path}/{save_path}", "w"), indent=2)


def basic_profile(args: argparse.Namespace, llm_agent: Agent):
    """
    In this function, we would like to summarize the client's basic profile.
    """

    # load the conversation
    conv_path_list = load_corpus(args.conv_path)
    # load the extracted_info content
    extracted_info = json.load(open(f"simulated_content/profile.json", "r", encoding="utf-8"))
    system_prompt = extracted_info.get("system_prompt", "")
    queries = extracted_info["queries"]

    # prepare the output path
    model_name = "claude"
    if "gpt" in args.model_name:
        model_name = "gpt"
    elif "llama" in args.model_name.lower():
        model_name = "llama"
    elif "mistral" in args.model_name.lower():
        model_name = "mistral"
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
    else:
        dataset = args.conv_path.split("/")[-1]
        args.output_path = f"output/{model_name}_annotated/{dataset}/profile/"
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)

    # iterate over the conversations

    if os.path.exists(args.model_name):
        # Define the batch size
        batch_size = 8
        # Loop over the conv_path_list in batches
        for i in tqdm(range(0, len(conv_path_list), batch_size),
                      total=(len(conv_path_list) + batch_size - 1) // batch_size):
            conv_path_batch = conv_path_list[i:i + batch_size]
            save_path_batch = [conv_path.split("/")[-1].replace(".txt", ".json") for conv_path in conv_path_batch]
            for path_idx, (save_path, conv_path) in enumerate(zip(save_path_batch, conv_path_batch)):
                if os.path.exists(f"{args.output_path}/{save_path}"):
                    conv_path_batch[path_idx] = ""
                    save_path_batch[path_idx] = ""
            conv_path_batch = [conv_path for conv_path in conv_path_batch if conv_path]
            save_path_batch = [save_path for save_path in save_path_batch if save_path]
            name_batch = ["Not Specified" for _ in range(len(save_path_batch))]
            assert len(conv_path_batch) == len(name_batch) == len(save_path_batch)
            if len(save_path_batch) == 0:
                continue
            conv_batch = [open(conv_path, "r").readlines()[:240] for conv_path in conv_path_batch]
            conv_string_batch = [conv_list2str(conv) for conv in conv_batch]
            # expected result list
            answer_json_batch = [{} for _ in range(len(save_path_batch))]
            for info, query in queries.items():
                query_batch = [re.sub("the client", name, query, flags=re.IGNORECASE) for name in name_batch]
                content_batch = [system_prompt.format(conversation=conv_string, query=query) for conv_string, query in
                                 zip(conv_string_batch, query_batch)]
                query_message_batch = [Message(agent_name="user", content=content, turn=1) for content in content_batch]
                response_batch = llm_response(llm_agent, query_message_batch)
                if isinstance(response_batch, str):
                    response_batch = [response_batch]
                response_batch = [response.replace("\n\n", ". ").replace("..", ".") for response in response_batch]
                assert len(response_batch) == len(save_path_batch)
                for idx, response in enumerate(response_batch):
                    answer_json_batch[idx][info] = response
                    if info == "name":
                        name_batch[idx] = response
            for idx, save_path in enumerate(save_path_batch):
                assert answer_json_batch[idx] != {}
                json.dump(answer_json_batch[idx], open(f"{args.output_path}/{save_path}", "w"), indent=2)

    else:
        name = "Not Specified"
        for conv_idx, conv_path in tqdm(enumerate(conv_path_list), total=len(conv_path_list)):
            save_path = conv_path.split("/")[-1].replace(".txt", ".json")
            if os.path.exists(f"{args.output_path}/{save_path}"):
                answer_json = json.load(open(f"{args.output_path}/{save_path}", "r"))
                is_break = True
                for key, content in answer_json.items():
                    if content == "" or "<<<<<<END_OF_CONVERSATION>>>>>>" in content:
                        is_break = False
                        break
                if is_break:
                    continue
                logging.info(f"Rewriting {save_path}")
            else:
                answer_json = defaultdict(dict)
            conv = open(conv_path, "r").readlines()
            conv_string = conv_list2str(conv)
            if len(conv_string.split(" ")) > 2800:
                conv_string = " ".join(conv_string.split(" ")[:2800]).replace("\n ", "\n")

            # expected result list
            for info, query in queries.items():
                if "Not Specified" not in name:
                    query = re.sub("the client", name, query, flags=re.IGNORECASE)
                content = system_prompt.format(conversation=conv_string, query=query)
                query_message = Message(agent_name="user", content=content, turn=1)
                # obtain answer
                response = answer_json.get(info, "")
                if response == "" or "<<<<<<END_OF_CONVERSATION>>>>>>" in response:
                    response = llm_response(llm_agent, query_message).replace("\n\n", ". ").replace("..", ".")
                    answer_json[info] = response.replace("\n", " ").strip()
                if info == "name":
                    name = response

            # save the answer
            json.dump(answer_json, open(f"{args.output_path}/{save_path}", "w"), indent=2)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    llm_agent = Agent(temperature=0, model=args.model_name, max_tokens=256)
    if args.extracted_info == "all":
        output_path = args.output_path
        args.output_path = f"{output_path}/profile/"
        basic_profile(args, llm_agent)
        args.output_path = f"{output_path}/big_five/"
        big_five_personality_traits(args, llm_agent)
        args.output_path = f"{output_path}/symptoms/"
        symptom_identification(args, llm_agent)
    elif args.extracted_info == "profile":
        basic_profile(args, llm_agent)
    elif args.extracted_info == "big_five":
        big_five_personality_traits(args, llm_agent)
    else:
        symptom_identification(args, llm_agent)
