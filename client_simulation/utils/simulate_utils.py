import logging
import os
import json
from tqdm import tqdm
from typing import List, Dict, Union
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

logging.getLogger().setLevel(logging.INFO)

STOP_WORDS = [
    'summary', 'one', 'sentence', 'situation', 'is', 'of', 'the', 'mental', 'and', 'the', 'current', 'health',
    'client', "feelings", "emotions", "therapist", 'his', 'her', 'he', 'she', 'they', 'them', 'their', 'their',
]

TOPICS = ["smoking cessation", "alcohol consumption", "substance abuse", "weight management", "medication adherence",
          "recidivism", "others"]


def obtain_symptoms(data_dir: Union[str, List[str]]):
    all_symptoms = [
        'Neurodevelopmental Disorders', 'Schizophrenia Spectrum and Other Psychotic Disorders',
        'Bipolar and Related Disorders', 'Depressive Disorders', 'Anxiety Disorders',
        'Obsessive-Compulsive and Related Disorders', 'Trauma- and Stressor-Related Disorders',
        'Dissociative Disorders', 'Somatic Symptom and Related Disorders', 'Feeding and Eating Disorders',
        'Elimination Disorders', 'Sleep-Wake Disorders', 'Sexual Dysfunctions', 'Gender Dysphoria',
        'Disruptive, Impulse-Control, and Conduct Disorders', 'Substance-Related and Addictive Disorders',
        'Neurocognitive Disorders', 'Personality Disorders', 'Paraphilic Disorders', 'Other Mental Disorders',
        'Medication-Induced Movement Disorders and Other Adverse Effects of Medication', 'Others'
    ]
    # obtain all the json file paths
    if isinstance(data_dir, str):
        file_list = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]
    else:
        file_list = []
        for dir_path in data_dir:
            file_list.extend([os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)])
    is_json = [file_name for file_name in file_list if file_name.endswith(".json")]
    assert all(is_json), f"All files should be in json format."

    all_symptom_types = defaultdict(list)
    all_symptom_description = {}
    all_problem_feelings = {}
    for file_path in tqdm(file_list, total=len(file_list)):
        # load the json file
        symptom_list = []
        data = json.load(open(file_path, "r", encoding="utf-8"))
        problem_type = data["problem type"]
        problem_detection = data["problem detection"]
        one_sentence_summary = data["one sentence summary"]
        for symptom in all_symptoms:
            if symptom in problem_type:
                symptom_list.append(symptom)
        data["symptoms"] = symptom_list
        json.dump(data, open(file_path, "w"), indent=2)
        file_index = file_path.split("/")[-3] + " " + file_path.split("/")[-1].split(".")[0]
        for symptom in symptom_list:
            all_symptom_types[symptom].append(file_index)
        all_symptom_description[file_index] = problem_detection.replace("\n", " ")
        all_problem_feelings[file_index] = one_sentence_summary.replace("\n", " ")

    if not os.path.exists("../analyses_content/"):
        os.makedirs("../analyses_content/")
    json.dump(all_symptom_types, open("../analyses_content/all_symptom_types.json", "w"), indent=2)
    json.dump(all_symptom_description, open("../analyses_content/all_symptom_description.json", "w"), indent=2)
    json.dump(all_problem_feelings, open("../analyses_content/all_problem_feelings.json", "w"), indent=2)


def cluster_text(key2documents, num_clusters, save_path):
    path_document_tuples = [(key, line.strip().lower()) for key, line in key2documents.items()]
    documents = [line for key, line in path_document_tuples]
    keys = [key for key, line in path_document_tuples]
    # Convert text documents to TF-IDF features
    default_stop_words = TfidfVectorizer(stop_words="english").get_stop_words()
    stop_words = list(default_stop_words) + STOP_WORDS

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(documents)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    kmeans.fit(X)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Initialize a dictionary to store the instances in each cluster
    cluster_instances = defaultdict(list)

    # Iterate over each document and its corresponding cluster label
    for instance, label in enumerate(cluster_labels):
        cluster_instances[str(label)].append(keys[instance])
    # Save the clustering results
    json.dump(cluster_instances, open(save_path.replace(".txt", ".json"), "w"), indent=2)

    # Save the clustering results
    with open(save_path, "w") as f:
        terms = vectorizer.get_feature_names_out()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        for i in range(num_clusters):
            f.write("Cluster {}:\n".format(i))
            # for j in order_centroids[i, :10]:  # Print top 10 most frequent words
            top_words = [terms[ind] for ind in order_centroids[i, :20]]
            f.write(", ".join(top_words))
            f.write("\n\n")


def cluster_symptoms(num_clusters: int = 7):
    assert os.path.exists(
        "../analyses_content/all_symptom_types.json"), f"Data directory all_symptoms.txt does not exist."

    all_symptom_description = json.load(open("../analyses_content/all_symptom_description.json", "r"))
    all_problem_feelings = json.load(open("../analyses_content/all_problem_feelings.json", "r"))

    # Convert text documents to TF-IDF features
    cluster_text(all_symptom_description, num_clusters, "../analyses_content/symptom_description_cluster.txt")
    cluster_text(all_problem_feelings, num_clusters, "../analyses_content/problem_feelings_cluster.txt")


def check_cluster_symptoms():
    cluster2symptom = json.load(open("../analyses_content/all_symptom_types.json", "r"))
    symptom2cluster = defaultdict(list)
    for key, value in cluster2symptom.items():
        for path in value:
            symptom2cluster[path].append(key)

    cluster2description = json.load(open("../analyses_content/symptom_description_cluster.json", "r"))
    description2cluster = {}
    for key, value in cluster2description.items():
        for path in value:
            description2cluster[path] = key

    cluster2feelings = json.load(open("../analyses_content/problem_feelings_cluster.json", "r"))
    feelings2cluster = {}
    for key, value in cluster2feelings.items():
        for path in value:
            feelings2cluster[path] = key

    conv2clusters = {}
    for key, value in symptom2cluster.items():
        conv2clusters[key] = (value, description2cluster[key], feelings2cluster[key])

    json.dump(conv2clusters, open("../analyses_content/clusters.json", "w"), indent=2)


if __name__ == "__main__":
    # obtain_symptoms(["../output/AnnoMI_transcript/formatted_profile", "../output/HighLow_transcript/formatted_profile"])
    cluster_symptoms()
    check_cluster_symptoms()
