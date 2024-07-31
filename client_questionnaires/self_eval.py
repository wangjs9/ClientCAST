from typing import List, Dict, Union
import argparse
import os
import logging

args = argparse.ArgumentParser()
args.add_argument("--openai_api_key", type=str, default="")
args.add_argument("--openai_api_base", type=str, default="")
args.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
args.add_argument("--questionnaire_name", type=str, default="")
args.add_argument("--data_dir", type=str, default="")
args = args.parse_args()

os.environ["OPENAI_API_KEY"] = args.openai_api_key
os.environ["OPENAI_API_BASE"] = args.openai_api_base

logging.getLogger().setLevel(logging.INFO)


def evaluation_before_counselling(data_path: Union[Dict[str], str], questionnaire_name: str):
    if type(data_path) == dict:
        data_path = [data_path]
    for path in data_path:
        pass
    return None


def evaluation_after_counselling():
    return None


if __name__ == "__main__":
    """
    This script is used to evaluate the performance of the client agent.
    There are twice evaluations: before the counselling and after the counselling.
    """
    # before the counselling
    evaluation_before_counselling(args.data_dir, args.questionnaire_name)
    # after the counselling
