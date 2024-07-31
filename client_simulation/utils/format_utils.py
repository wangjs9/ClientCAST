import os
import re
from typing import List, Union
import json
import argparse
from tqdm import tqdm
import logging

logging.getLogger().setLevel(logging.INFO)

SPEAKER = {"T": "Therapist", "C": "Client"}
PRONOUN = {"his": "he", "her": "she", "their": "they"}


def load_corpus(data_path: str) -> List[str]:
    """
    Load the corpus from the given data path.

    Args:
    data_path: Union[Dict[str], str], the data path

    Returns:
    corpus_list: List[str], the list of conversation strings
    """
    assert os.path.exists(data_path), f"Data path {data_path} does not exist."
    if os.path.isdir(data_path):
        corpus_list = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".txt")]
    else:
        corpus_list = [data_path]
    return sorted(corpus_list)


def conv_list2str(conv_list: List[str]) -> str:
    """
    Given a list of conversation strings, convert them into a single conversation string.

    Args:
    conv_list: List[str], a list of conversation strings

    Returns:
    conv_string: str, a single conversation string
    """
    new_conv_list = []
    for turn in conv_list:
        turn = turn.strip()
        role = SPEAKER[turn.split(": ")[0]]
        utterance = turn[3:]
        new_conv_list.append((role, utterance))

    # obtain conversation
    conv_string = ""
    conv_length = len(new_conv_list)
    for conv_idx, (role, utterance) in enumerate(new_conv_list):
        conv_string += f"{role}: {utterance}"
        if conv_idx < conv_length - 1:
            conv_string += "\n"

    return conv_string


def adjust_and_remove(text, match):
    start, end = match.span()

    # Determine if there are spaces before and after the match
    space_before = start > 0 and text[start - 1] == ' '
    space_after = end < len(text) and text[end] == ' '

    # Adjust the start index to remove the space before the pattern
    if space_before:
        start -= 1

    # Adjust the end index to remove the space after the pattern
    if space_after:
        end += 1

    # Check if both before and after spaces exist; adjust to leave one space if so
    if space_before and space_after:
        text = text[:start + 1] + text[end:]  # Leave one space
    else:
        text = text[:start] + text[end:]  # Remove the space entirely

    return text.strip()  # Strip to clean up any leading/trailing spaces


def remove_pattern(text, pattern):
    # Use a compiled regular expression to improve performance for repeated calls
    regex = re.compile(pattern, re.IGNORECASE)

    # Search for the pattern in the text
    while True:
        match = regex.search(text)
        if not match:
            break
        text = adjust_and_remove(text, match)
    return text.strip()  # If no pattern found, return the original text stripped of any extra spaces


def extract_first_sentence(text):
    # Find the index of the first period followed by a space or the end of the string
    end_index = text.find('. ')
    # If a period followed by a space is found, adjust the index to include the period
    if end_index != -1:
        return text[:end_index + 1]
    # If no period followed by a space is found, check if the text ends with a period
    if text.endswith('.'):
        return text
    # Return None if no sentence-ending period is found
    else:
        return text


def replace_pronouns_and_adjust_verbs(text, name, pronoun=None):
    # Define a regular expression pattern that matches the name or pronoun followed by any verb
    if pronoun is None:
        pattern = r"\b" + re.escape(name) + r"'s\b|\b" + re.escape(name) + r"\s+(\w+)\b"
        possessive_pronoun, reflexive_pronoun, objective_pronoun = None, None, None
    else:
        possessive_pronoun = "her" if pronoun == "she" else "his"
        if pronoun == "they":
            possessive_pronoun = "their"
        reflexive_pronoun = "herself" if pronoun == "she" else "himself"
        if pronoun == "they":
            reflexive_pronoun = "themselves"
        objective_pronoun = "her" if pronoun == "she" else "him"
        if pronoun == "they":
            objective_pronoun = "them"

        pattern = (
                r"\b" + re.escape(name) + r"'s\b|" +
                r"\b" + re.escape(name) + r"\b(?:\s+(\w+))?" +
                r"|\b" + pronoun + r"\b\s+(\w+)" +
                r"|\b" + possessive_pronoun + r"\b" +
                r"|\b" + reflexive_pronoun + r"\b" +
                r"|\b" + objective_pronoun + r"\b"
        )

    # Replacement function to adjust verbs and possessives
    def custom_replacer(match):
        # Check for possessive and reflexive forms
        word_matched = match.group(0).lower()
        if word_matched == f"{name.lower()}'s" or (
                possessive_pronoun is not None and word_matched == possessive_pronoun):
            return "your"
        elif reflexive_pronoun is not None and word_matched == reflexive_pronoun:
            return "yourself"
        elif objective_pronoun is not None and word_matched == objective_pronoun:
            return "you"
        # Check for verb after the name or pronoun
        verb = match.group(1) or match.group(2)
        if verb:
            if verb == "is":
                return "you are"
            elif verb == "was":
                return "you were"
            elif verb == "has":
                return "you have"
            else:
                # Attempt to conjugate the verb correctly for "you"
                return "you " + re.sub(r"([^aeiou])ies$", r"\1y", re.sub(r"([^s])s$", r"\1", verb))
        # If no verb follows directly, replace the name or pronoun with 'you'
        return "you"

    # Replace the pattern in the text using the replacement function, handling case insensitivity
    return re.sub(pattern, custom_replacer, text, flags=re.IGNORECASE)


def find_first_word_after_substring(text, start_substring, end_substring):
    # Pattern to match the start substring, optional content (non-greedy), and end substring followed by capturing the first word
    pattern = re.escape(start_substring) + r'.*?' + re.escape(end_substring) + r'\s+(\w+)'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # If a match is found, return the captured word
    if match:
        return match.group(1)  # The first group captures the word after the end substring
    return None


def remove_sentence_with_phrase(text, phrase):
    # Define a pattern that matches a sentence containing the specified phrase
    # The sentence starts at the beginning of the string or after a period and ends with a period
    pattern = re.compile(r'(?:(?<=\.)\s*|^)[^.]*?\b' + re.escape(phrase) + r'\b[^.]*\.', re.IGNORECASE)

    # Replace the matched sentences with an empty string
    cleaned_text = re.sub(pattern, '', text)

    # Strip and correct spaces where sentences were removed
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()

    return cleaned_text


def capitalize_sentences(text):
    # Function to capitalize the first letter of a sentence
    def capitalize_match(match):
        # Capitalizes the first non-space character after a period
        return match.group(1) + match.group(2).upper()

    # Capitalize the first character of the text if it's lowercase
    if text:
        text = text[0].upper() + text[1:]

    # Regex to find sentence boundaries and the first character of the next sentence
    # This pattern assumes sentences end with a period followed by a space, then starts the next sentence
    text = re.sub(r'(\.\s+)([a-z])', capitalize_match, text)

    return text


def format_profile(data_dir: Union[str, List[str]]):
    logging.info(f"Formatting the profile data.")

    if isinstance(data_dir, str):
        file_path_list = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]
        data_dir = [data_dir]
    else:
        file_path_list = []
        for dir_path in data_dir:
            file_path_list.extend([os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)])
    is_json = [file_name for file_name in file_path_list if file_name.endswith(".json")]
    assert all(is_json), f"All files should be in json format."

    for dir_path in data_dir:
        if not os.path.exists(dir_path.replace("profile", "profile_formatted")):
            os.makedirs(dir_path.replace("profile", "profile_formatted"))

    for file_path in tqdm(sorted(file_path_list), total=len(file_path_list)):
        save_path = file_path.replace("profile", "profile_formatted")
        if os.path.exists(save_path):
            continue
        data = json.load(open(file_path, "r", encoding="utf-8"))
        new_data = {}

        name = data["name"].strip(".\n")
        if name != "Not Specified":
            new_data["name"] = name
        else:
            name = "the client"

        gender = data["gender"].strip(".\n")
        pronoun = None
        if gender != "Cannot be identified":
            new_data["gender"] = gender
            pronoun = "she" if gender == "Female" else "he"

        age = extract_first_sentence(data["age"]).strip(".\n")
        if not age.startswith("Unclear"):
            age = age.replace("Estimated age: ", "")
            age = remove_pattern(age, r", based on.*$")
            age = replace_pronouns_and_adjust_verbs(age, name, pronoun)
            new_data["age"] = age.strip().lower()

        occupation = data["occupation"].strip(".\n")
        if occupation != "Not Specified":
            occupation = extract_first_sentence(occupation.lower())
            occupation = replace_pronouns_and_adjust_verbs(occupation, name, pronoun)
            new_data["occupation"] = occupation

        if "name" in new_data:
            reasons = re.sub("the client", name, data["reasons"], flags=re.IGNORECASE)
        else:
            reasons = data["reasons"]
        if name == "the client" and pronoun is None:
            pronoun = find_first_word_after_substring(
                reasons.lower(), "the client is visiting", "the therapist because")
            pronoun = PRONOUN.get(pronoun, pronoun)
            if pronoun not in ["he", "she", "they"]:
                pronoun = find_first_word_after_substring(
                    reasons.lower(), "the client is visiting", "the therapist because of concerns about")
            pronoun = PRONOUN.get(pronoun, pronoun)
            # if pronoun:
            #     assert pronoun in ["he", "she", "they"], f"Pronoun {pronoun} not found."
            if pronoun in ["he", "she", "they"]:
                reasons = replace_pronouns_and_adjust_verbs(reasons, name, pronoun)
        else:
            reasons = replace_pronouns_and_adjust_verbs(reasons, name, pronoun)
        new_data["reasons"] = capitalize_sentences(reasons)

        if "name" in new_data:
            topic = re.sub("the client", name, data["topic"], flags=re.IGNORECASE)
        else:
            topic = data["topic"]
        new_data["topic"] = topic

        if "name" in new_data:
            situation = re.sub("the client", name, data["problem"], flags=re.IGNORECASE)
        else:
            situation = data["problem"]
        situation = replace_pronouns_and_adjust_verbs(situation, name, pronoun)
        new_data["situation"] = capitalize_sentences(situation)

        if "name" in new_data:
            problem = re.sub("the client", name, data["situation"], flags=re.IGNORECASE)
        else:
            problem = data["problem"]
        problem = replace_pronouns_and_adjust_verbs(problem, name, pronoun)
        new_data["problem"] = capitalize_sentences(problem)

        if "name" in new_data:
            emotion = re.sub("the client", name, data["emotion"], flags=re.IGNORECASE)
        else:
            emotion = data["emotion"]
        emotion = replace_pronouns_and_adjust_verbs(emotion, name, pronoun)
        new_data["emotion"] = capitalize_sentences(emotion)

        if "name" in new_data:
            feeling_expression = re.sub(
                "the client", name, data["feeling expression"].strip(), flags=re.IGNORECASE)
        else:
            feeling_expression = data["feeling expression"].strip()
        if not feeling_expression.startswith("Cannot be identified"):
            feeling_expression = replace_pronouns_and_adjust_verbs(feeling_expression, name, pronoun)
            if feeling_expression:
                new_data["feeling expression"] = "Unwillingness to express emotion is " + capitalize_sentences(
                    feeling_expression)

        if "name" in new_data:
            emotional_fluctuation = re.sub(
                "the client", name, data["emotional fluctuation"].strip(), flags=re.IGNORECASE)
        else:
            emotional_fluctuation = data["emotional fluctuation"].strip()
        if not emotional_fluctuation.startswith("Cannot be identified"):
            emotional_fluctuation = replace_pronouns_and_adjust_verbs(emotional_fluctuation, name, pronoun)
            if emotional_fluctuation:
                new_data["emotional fluctuation"] = "Emotional fluctuation is " + capitalize_sentences(
                    emotional_fluctuation)

        if "name" in new_data:
            resistance = re.sub("the client", name, data["resistance"].strip(), flags=re.IGNORECASE)
        else:
            resistance = data["resistance"].strip()
        if not resistance.startswith("Cannot be identified"):
            resistance = replace_pronouns_and_adjust_verbs(resistance, name, pronoun)
            if resistance:
                new_data["resistance"] = "Resistance towards the therapist is " + capitalize_sentences(resistance)

        json.dump(new_data, open(save_path, "w"), indent=2)


def format_big_five(data_dir: Union[str, List[str]]):
    logging.info(f"Formatting the personality data.")

    if isinstance(data_dir, str):
        file_path_list = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]
        data_dir = [data_dir]
    else:
        file_path_list = []
        for dir_path in data_dir:
            file_path_list.extend([os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)])
    is_json = [file_name for file_name in file_path_list if file_name.endswith(".json")]
    assert all(is_json), f"All files should be in json format."

    for dir_path in data_dir:
        if not os.path.exists(dir_path.replace("big_five", "big_five_formatted")):
            os.makedirs(dir_path.replace("big_five", "big_five_formatted"))

    for file_path in tqdm(sorted(file_path_list), total=len(file_path_list)):
        save_path = file_path.replace("big_five", "big_five_formatted")
        if os.path.exists(save_path):
            continue
        profile_path = save_path.replace("big_five", "profile")
        profile = json.load(open(profile_path, "r", encoding="utf-8"))
        name = profile.get("name", "the client")
        gender = profile.get("gender", None)
        pronoun = "she" if gender == "Female" else "he"
        if gender is None:
            pronoun = "they"

        data = json.load(open(file_path, "r", encoding="utf-8"))
        new_data = {}

        for trait, trait_content in data.items():
            trait_content = replace_pronouns_and_adjust_verbs(trait_content, name, pronoun)
            trait_content = replace_pronouns_and_adjust_verbs(trait_content, "the client", pronoun)
            trait_content = replace_pronouns_and_adjust_verbs(trait_content, name, "they")
            for verb in [" demonstrate ", " exhibit "]:
                trait_content = trait_content.replace(verb, f" can{verb}")

            trait_content = trait_content.replace(" seem to ", " can appear to ")
            trait_content = capitalize_sentences(trait_content)
            new_data[trait] = trait_content

        json.dump(new_data, open(save_path, "w"), indent=2)


def rank_keyword():
    pass


def format_symptoms(data_dir: Union[str, List[str]]):
    def extract_severity(text):
        text = text.replace("**", "").replace("...", "").strip()
        pattern = r"The severity[^.]*is approximately[^.]*\."
        extraction = [match.group(0) for match in re.finditer(pattern, text, re.IGNORECASE)]
        assert len(extraction) > 0, f"Multiple matches found for severity: {text}"
        extraction = extraction[0]

        pattern = r'\d+\.\d+|\d+'
        numbers = re.findall(pattern, extraction)
        assert len(numbers) == 1, f"Multiple numbers found in the text: {extraction}"
        explanation = re.sub(extraction, "", text, 1).strip()

        return numbers[0], explanation

    def process_explanation(explanation, name, pronoun):
        explanation = remove_pattern(explanation, r"based on the conversation(,)?")
        explanation = remove_pattern(explanation, r"this suggests (that)?")
        explanation = replace_pronouns_and_adjust_verbs(explanation, name, pronoun)
        explanation = replace_pronouns_and_adjust_verbs(explanation, "the client", pronoun)
        explanation = replace_pronouns_and_adjust_verbs(explanation, name, "they")
        explanation = explanation.replace(" expresse ", " have ")
        explanation = explanation.replace(" mention ", " can mention ")
        explanation = explanation.replace(" exhibit ", " can exhibit ")
        explanation = explanation.replace(" seem to ", " can appear to ")
        explanation = capitalize_sentences(explanation)
        return explanation

    logging.info(f"Formatting the symptoms data.")

    if isinstance(data_dir, str):
        file_path_list = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]
        data_dir = [data_dir]
    else:
        file_path_list = []
        for dir_path in data_dir:
            file_path_list.extend([os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)])
    is_json = [file_name for file_name in file_path_list if file_name.endswith(".json")]
    assert all(is_json), f"All files should be in json format."

    symptom_list = json.load(open("../simulated_content/symptoms.json", "r"))
    PHQ9_queries = symptom_list["PHQ-9"]["queries"]
    PHQ9_severity = symptom_list["PHQ-9"]["severity"]

    GAD7_queries = symptom_list["GAD-7"]["queries"]
    GAD7_severity = symptom_list["GAD-7"]["severity"]

    OQ45_queries = symptom_list["OQ-45"]["queries"]
    OQ45_pos_index = symptom_list["OQ-45"]["positive queries"]
    OQ45_severity_pos = symptom_list["OQ-45"]["positive severity"]
    OQ45_severity_neg = symptom_list["OQ-45"]["negative severity"]

    for dir_path in data_dir:
        if not os.path.exists(dir_path.replace("symptoms", "symptoms_formatted").replace("_denoised", "")):
            os.makedirs(dir_path.replace("symptoms", "symptoms_formatted").replace("_denoised", ""))

    for file_path in tqdm(sorted(file_path_list), total=len(file_path_list)):
        save_path = file_path.replace("symptoms", "symptoms_formatted")
        save_path = save_path.replace("_denoised", "")
        profile_path = save_path.replace("symptoms", "profile")
        profile = json.load(open(profile_path, "r", encoding="utf-8"))
        name = profile.get("name", "the client")
        gender = profile.get("gender", None)
        pronoun = "she" if gender == "Female" else "he"
        if gender is None:
            pronoun = "they"

        data = json.load(open(file_path, "r", encoding="utf-8"))
        new_data = []

        PHQ9 = data["PHQ-9"]
        for query_idx, symptom in PHQ9.items():
            if "Cannot be identified" in symptom:
                continue
            if symptom == " The severity is approximately ":
                continue
            symptom = symptom.replace("\n\n", ". ").replace(".. ", ". ")
            severity, explanation = extract_severity(symptom)
            explanation = process_explanation(explanation, name, pronoun)
            if PHQ9_severity[severity] != "Not at all" and explanation:
                new_data.append({
                    "symptom": capitalize_sentences(PHQ9_queries[query_idx]),
                    "severity": PHQ9_severity[severity],
                    "explanation": explanation
                })

        GAD7 = data["GAD-7"]
        for query_idx, symptom in GAD7.items():
            if "Cannot be identified" in symptom:
                continue
            if symptom == " The severity is approximately ":
                continue
            symptom = symptom.replace("\n\n", ". ").replace(".. ", ". ")
            severity, explanation = extract_severity(symptom)
            explanation = process_explanation(explanation, name, pronoun)
            if GAD7_severity[severity] != "Not at all" and explanation:
                new_data.append({
                    "symptom": capitalize_sentences(GAD7_queries[query_idx]),
                    "severity": GAD7_severity[severity],
                    "explanation": explanation
                })

        OQ45 = data["OQ-45"]
        for query_idx, symptom in OQ45.items():
            if "Cannot be identified" in symptom:
                continue
            if symptom == " The severity is approximately ":
                continue
            if query_idx in OQ45_pos_index:
                continue
            symptom = symptom.replace("\n\n", ". ").replace(".. ", ". ")
            severity, explanation = extract_severity(symptom)
            # severity = OQ45_severity_pos[severity]
            severity = OQ45_severity_neg[severity]
            explanation = process_explanation(explanation, name, pronoun)
            if severity != "Never" and explanation:
                new_data.append({
                    "symptom": capitalize_sentences(OQ45_queries[query_idx]),
                    "severity": severity,
                    "explanation": explanation
                })

        json.dump(new_data, open(save_path, "w"), indent=2)


def json2txt(data_dir: Union[str, List[str]], output_dir=None):
    logging.info(f"Converting the json data to txt format.")
    if output_dir is None:
        output_dir = data_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if isinstance(data_dir, str):
        file_path_list = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]
    else:
        file_path_list = []
        for dir_path in data_dir:
            file_path_list.extend([os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)])
    is_json = [file_name for file_name in file_path_list if file_name.endswith(".json")]
    assert all(is_json), f"All files should be in json format."

    for file_path in tqdm(sorted(file_path_list), total=len(file_path_list)):
        save_path = os.path.join(output_dir, file_path.split("/")[-1].replace(".json", ".txt"))
        data = json.load(open(file_path, "r", encoding="utf-8"))
        with open(save_path, "w", encoding="utf-8") as f:
            for line in data:
                speaker = line["agent_name"][0]
                content = line["content"]
                if content == "":
                    continue
                f.write(f"{speaker}: {content}\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--data_to_format', type=str, default='all', choices=['profile', 'big_five', 'symptoms'])
    args.add_argument('--json_to_txt', action='store_true')
    args.add_argument('--model_name', type=str, default='llama-70b')
    args.add_argument('--data_dir', type=str, default='LlamaC_vs_ClaudeT')
    args.add_argument('--data_directory_list', nargs='*', type=str)
    args = args.parse_args()

    if args.json_to_txt:
        # file_prefix = f"../../simulation_conversations/{args.data_dir}"
        file_prefix = f"../../human_interactions/{args.data_dir}"

        for data_name in ["HighLow", "AnnoMI"]:
            json2txt(os.path.join(file_prefix, f"{data_name}_transcript"),
                     os.path.join(file_prefix, f"{data_name}_{args.model_name}"))
    else:
        data_directory_list = args.data_directory_list
        if type(data_directory_list) == str:
            data_directory_list = [data_directory_list]
        # data_directory_list = [
        #     "../output/claude_annotated/HighLow_claude",
        #     "../output/claude_annotated/AnnoMI_transcript", "../output/claude_annotated/HighLow_transcript",
        #     "../output/gpt_annotated/AnnoMI_transcript", "../output/gpt_annotated/HighLow_transcript",
        #     "../output/llama-70b_annotated/AnnoMI_transcript", "../output/llama-70b_annotated/HighLow_transcript",
        #     "../output/Mixtral_annotated/AnnoMI_transcript", "../output/Mixtral_annotated/HighLow_transcript",
        #     "../output/llama-70b_annotated/AnnoMI_llama-70b", "../output/llama-70b_annotated/HighLow_llama-70b",
        #     "../output/Mixtral_annotated/AnnoMI_Mixtral", "../output/Mixtral_annotated/HighLow_Mixtral"
        # ]

        if args.data_to_format == "profile":
            data_directory_list = [os.path.join(directory, "profile") for directory in data_directory_list]
            format_profile(data_directory_list)
        elif args.data_to_format == "big_five":
            data_directory_list = [os.path.join(directory, "big_five") for directory in data_directory_list]
            format_big_five(data_directory_list)
        elif args.data_to_format == "symptoms":
            data_directory_list = [os.path.join(directory, "symptoms_denoised") for directory in data_directory_list]
            for idx, data_dir in enumerate(data_directory_list):
                if not os.path.exists(data_dir):
                    data_directory_list[idx] = data_directory_list[idx].replace("_denoised", "")
            format_symptoms(data_directory_list)
        elif args.data_to_format == "all":
            for data_to_format in ["profile", "big_five", "symptoms"]:
                if data_to_format == "profile":
                    data_directories = [os.path.join(directory, "profile") for directory in data_directory_list]
                    format_profile(data_directories)
                elif data_to_format == "big_five":
                    data_directories = [os.path.join(directory, "big_five") for directory in data_directory_list]
                    format_big_five(data_directories)
                elif data_to_format == "symptoms":
                    data_directories = [
                        os.path.join(directory, "symptoms_denoised") for directory in data_directory_list]
                    for idx, data_dir in enumerate(data_directories):
                        if not os.path.exists(data_dir):
                            data_directories[idx] = data_directories[idx].replace("_denoised", "")
                    format_symptoms(data_directories)
