import json
import time
from collections import defaultdict
from functools import wraps

import requests
import os
import numpy as np
import torch
from PIL import Image
from annoy import AnnoyIndex
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from util.text_description_set import aid_candidate_descriptions, siri_candidate_descriptions, \
    hellors_candidate_descriptions, nwpu_resisc45_candidate_descriptions, whu_rs19_candidate_descriptions, \
    patternnet_candidate_descriptions, rsc11_candidate_descriptions, dlrsd_candidate_descriptions

# 标签映射
label_to_id = {name: idx for idx, name in enumerate([
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral", "denseresidential",
    "forest", "freeway", "golfcourse", "harbor", "intersection", "mediumresidential",
    "mobilehomepark", "overpass", "parkinglot", "river", "runway", "sparseresidential",
    "storagetanks", "tenniscourt"
])}
id_to_label = {v: k for k, v in label_to_id.items()}

# 提前加载 SentenceTransformer
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def similarity_bert(sent1, sent2):
    embeddings = sentence_model.encode([sent1, sent2], convert_to_tensor=True)
    emb1, emb2 = embeddings[0].cpu().numpy().reshape(1, -1), embeddings[1].cpu().numpy().reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]

device = "cuda" if torch.cuda.is_available() else "cpu"
shared_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch14").to(device)
shared_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch14")

class ImageFeatureStore:
    def __init__(self, dim=512, n_trees=100):
        self.device = device
        self.model = shared_model
        self.processor = shared_processor
        self.dim = dim
        self.n_trees = n_trees
        self.index = None
        self.image_paths = []
        self.text_descriptions = []
        self.feature_cache = {}
        self.dataset_descriptions = {
            "AID": aid_candidate_descriptions,
            "SIRI-WHU": siri_candidate_descriptions,
            "HelloRS": hellors_candidate_descriptions,
            "NWPU-RESISC45": nwpu_resisc45_candidate_descriptions,
            "WHU-RS19": whu_rs19_candidate_descriptions,
            "PatternNet": patternnet_candidate_descriptions,
            "RSC11": rsc11_candidate_descriptions,
            "DLRSD": dlrsd_candidate_descriptions
        }

    def _normalize(self, vector):
        return vector / np.linalg.norm(vector)

    def extract_image_features(self, image_path):
        if image_path in self.feature_cache:
            return self.feature_cache[image_path]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        features = features.cpu().numpy().astype('float32')[0]
        normed = self._normalize(features)
        self.feature_cache[image_path] = normed
        return normed

    def build_index_from_datasets(self, dataset_root_dir):
        self.index = AnnoyIndex(self.dim, 'angular')
        item_count = 0
        for dataset_name in os.listdir(dataset_root_dir):
            dataset_path = os.path.join(dataset_root_dir, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            print(f"Processing dataset: {dataset_name}")
            descriptions = self.dataset_descriptions.get(dataset_name, [])
            for class_idx, class_name in enumerate(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                description = descriptions[class_idx]
                for img_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        features = self.extract_image_features(img_path)
                        self.index.add_item(item_count, features)
                        self.image_paths.append(img_path)
                        self.text_descriptions.append(description)
                        item_count += 1
                    except Exception:
                        continue
        print("Building index...")
        self.index.build(self.n_trees)
        print(f"Index built with {item_count} items")

    def save_index(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        if self.index:
            self.index.save(os.path.join(save_path, "index.ann"))
        with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "image_paths": self.image_paths,
                "text_descriptions": self.text_descriptions,
                "dim": self.dim,
                "n_trees": self.n_trees
            }, f)

    def load_index(self, load_path):
        with open(os.path.join(load_path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
            self.image_paths = metadata["image_paths"]
            self.text_descriptions = metadata["text_descriptions"]
            self.dim = metadata["dim"]
            self.n_trees = metadata["n_trees"]
        self.index = AnnoyIndex(self.dim, 'angular')
        self.index.load(os.path.join(load_path, "index.ann"))

    def search_similar_images(self, query_image_path, k=5, search_k=10000):
        query_vector = self.extract_image_features(query_image_path)
        indices, distances = self.index.get_nns_by_vector(query_vector, k, include_distances=True, search_k=search_k)
        results = []
        for idx, dist in zip(indices, distances):
            similarity = 1 - dist ** 2 / 2
            results.append({
                "similarity": similarity,
                "description": self.text_descriptions[idx],
                "image_path": self.image_paths[idx]
            })
        return results

def format_example(example):
    return f"Description: {example['description']}\nTag: {example['tags']}"

def find_most_similar_example(target_desc, examples):
    max_similarity = -1
    best_match = None
    for example in (examples if isinstance(examples, list) else [examples]):
        current_similarity = similarity_bert(target_desc, example["description"])
        if current_similarity > max_similarity:
            max_similarity = current_similarity
            best_match = example
    return best_match if max_similarity > 0.7 else None

def build_prompt(best_match_description, example_file_path):
    description_text = str(best_match_description)
    with open(example_file_path, 'r', encoding='utf-8') as f:
        example_data = json.load(f)
    matched_example = find_most_similar_example(description_text, example_data)
    category_mapping_str = "\n".join(f"{id}. {id_to_label[id]}" for id in sorted(id_to_label.keys()))
    example_str = format_example(matched_example) if matched_example else "No example available"
    final_prompt = f"""
    **Context**: You are analyzing openly available satellite images from a public research dataset.
    **Image Description**:
    {description_text}
    **Allowed Categories**:
    {category_mapping_str}
    **Instructions**:
    1. Select the single most relevant category NUMBER (0-20)
    2. Respond only with the number, no other text
    3. Example correct response: "0"
    **Example**:
    {example_str}
    **Your Response (number only)**:
    """
    return final_prompt

def count_tokens(text):
    return len(text.split())


perf_stats = defaultdict(list)
total_tokens = {'input': 0, 'output': 0}


def monitor_throughput(func):
    @wraps(func)
    def wrapper(prompt, *args, **kwargs):
        start_time = time.time()
        input_tokens = count_tokens(prompt)
        total_tokens['input'] += input_tokens
        response_text = func(prompt, *args, **kwargs)

        output_tokens = count_tokens(response_text)
        duration = time.time() - start_time
        total_tokens['output'] += output_tokens
        perf_stats['latency'].append(duration)

        return response_text

    return wrapper

@monitor_throughput
def llm_generate_labels_qwen(prompt, model='qwen2.5:1.5b'):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 128
        }
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get('response', '')
    except Exception as e:
        print(f"LLM call exception: {e}")
        return ""

@monitor_throughput
def llm_generate_labels_llama(prompt, model="llama3.2:1b"):
    messages = [
        {"role": "user", "content": prompt}
    ]
    url = "http://localhost:11434/api/chat"  # 注意改为/chat端点
    data = {
        "model": model,
        "messages": messages,  # 使用消息列表
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512
        }
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['message']['content']
    else:
        raise Exception(f"Request failed, status code: {response.status_code}, Response content: {response.text}")

@monitor_throughput
def llm_generate_labels_granite3(prompt, model="granite3-moe:latest"):
    messages = [
        {"role": "user", "content": prompt}
    ]
    url = "http://localhost:11434/api/chat"
    data = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512
        }
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['message']['content']
    else:
        raise Exception(f"Request failed, status code: {response.status_code}, Response content: {response.text}")


def validate_labels(labels):
    valid_labels = []
    labels = labels.strip()
    if labels.startswith("Tag: "):
        labels = labels[5:].strip()
    for label in labels.split(","):
        label = label.strip()
        if label.isdigit() and int(label) in label_to_id.values():
            valid_labels.append(int(label))
    return valid_labels

def generate_labels_with_retry(model_name, prompt, max_retries=1):
    global labels
    for _ in range(max_retries):
        if model_name == 'qwen':
            labels = llm_generate_labels_qwen(prompt)
        elif model_name == 'llama':
            labels = llm_generate_labels_llama(prompt)
        elif model_name == 'granite3':
            labels = llm_generate_labels_granite3(prompt)
        valid_labels = validate_labels(labels)
        if valid_labels:
            return valid_labels
    return []

def generate_tags_for_dataset(model_name,knowledge_base):
    tagged_knowledge_base = []
    feature_store = ImageFeatureStore()
    feature_store.load_index("../vector_db/annoy_index")
    for idx, item in enumerate(knowledge_base):
        query_img = item["metadata"]["file_path"]
        similar_results = feature_store.search_similar_images(query_img, k=3)
        best_match_description = max(similar_results, key=lambda x: x['similarity'])['description']

        prompt = build_prompt(best_match_description, '../rag/ucm/label.json')
        valid_labels = generate_labels_with_retry(model_name,prompt)
        item["generated_tags"] = ", ".join(map(str, valid_labels))
        tagged_knowledge_base.append(item)

        if (idx + 1) % 10 == 0 or idx == len(knowledge_base) - 1:
            print(f"Processed {idx + 1}/{len(knowledge_base)}")

    return tagged_knowledge_base

if __name__ == "__main__":
    with open("../rag/ucm/knowledge_base_clip_ucm.json", "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
    model_name = 'qwen'
    tagged_knowledge_base = generate_tags_for_dataset(model_name,knowledge_base)
    os.makedirs("../result/ucm/first_tag", exist_ok=True)
    with open("..result/ucm/first_tag/first_tag_qwen_ucm.json", "w", encoding="utf-8") as f:
        json.dump(tagged_knowledge_base, f, ensure_ascii=False, indent=4)

