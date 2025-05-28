import os
import numpy as np
import torch
from PIL import Image
from annoy import AnnoyIndex
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pickle

from util.text_description_set import aid_candidate_descriptions, nwpu_resisc45_candidate_descriptions, \
    whu_rs19_candidate_descriptions, patternnet_candidate_descriptions, rsc11_candidate_descriptions, \
    dlrsd_candidate_descriptions, hellors_candidate_descriptions, siri_candidate_descriptions


class ImageFeatureStore:
    def __init__(self, model_name="openai/clip-vit-base-patch14", dim=512, n_trees=100):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = dim
        self.n_trees = n_trees
        self.index = None
        self.image_paths = []
        self.text_descriptions = []
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
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        features = image_features.cpu().numpy().astype('float32')[0]
        return self._normalize(features)

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
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

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
        print(f"Index saved to {save_path}")

    def load_index(self, load_path):
        metadata_path = os.path.join(load_path, "metadata.pkl")
        index_path = os.path.join(load_path, "index.ann")

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            self.image_paths = metadata["image_paths"]
            self.text_descriptions = metadata["text_descriptions"]
            self.dim = metadata["dim"]
            self.n_trees = metadata["n_trees"]

        self.index = AnnoyIndex(self.dim, 'angular')
        self.index.load(index_path)
        print(f"Index loaded from {load_path}")

    def search_similar_images(self, query_image_path, k=5):
        query_vector = self.extract_image_features(query_image_path)
        indices, distances = self.index.get_nns_by_vector(
                query_vector, k, include_distances=True
            )

        results = []
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            similarity = 1 - dist ** 2 / 2
            results.append({
                    "rank": i + 1,
                    "similarity": similarity,
                    "description": self.text_descriptions[idx],
                    "image_path": self.image_paths[idx]
                })
        return results

