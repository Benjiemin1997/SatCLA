import os
import torch
import clip
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json
from collections import defaultdict

ucm_label_to_id = {
    "agricultural": 0,
    "airplane": 1,
    "baseballdiamond": 2,
    "beach": 3,
    "buildings": 4,
    "chaparral": 5,
    "denseresidential": 6,
    "forest": 7,
    "freeway": 8,
    "golfcourse": 9,
    "harbor": 10,
    "intersection": 11,
    "mediumresidential": 12,
    "mobilehomepark": 13,
    "overpass": 14,
    "parkinglot": 15,
    "river": 16,
    "runway": 17,
    "sparseresidential": 18,
    "storagetanks": 19,
    "tenniscourt": 20
}

rsi_label_to_id = {
    "cloud": 0,
    "desert": 1,
    "greenarea": 2,
    "water": 3
}

rs2800_label_to_id = {
    "farmland": 0,
    "forest": 1,
    "grassland": 2,
    "industry": 3,
    "parkinglot": 4,
    "resident": 5,
    "river": 6,
}

dataset_name = 'UCM'
label_to_id = []
if dataset_name == 'UCM':
    label_to_id = ucm_label_to_id
elif dataset_name == 'RSI':
    label_to_id = rsi_label_to_id
elif dataset_name == 'RS':
    label_to_id = rs2800_label_to_id
id_to_label = {v: k for k, v in label_to_id.items()}


class UCMercedLandDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.transform = transform
        self.paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = []
        self.image_metadata = []

        with open(json_path, "r", encoding="utf-8") as f:
            self.image_metadata = json.load(f)

        class_names = sorted(set(item["class_name"] for item in self.image_metadata))
        self.label_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.idx_to_label = class_names

        for img_data in self.image_metadata:
            file_path = os.path.normpath(img_data["metadata"]["file_path"])
            if not os.path.isabs(file_path):
                file_path = os.path.abspath(file_path)
            self.paths.append(file_path)
            self.labels.append(self.label_to_idx[img_data["class_name"]])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        file_path = self.paths[idx]

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        img = Image.open(file_path).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, idx

    def get_filename(self, idx):
        return os.path.basename(self.paths[idx])


class RS2800Dataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.transform = transform
        self.paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = []
        self.image_metadata = []

        with open(json_path, "r", encoding="utf-8") as f:
            self.image_metadata = json.load(f)

        class_names = sorted(set(item["class_name"] for item in self.image_metadata))
        self.label_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.idx_to_label = class_names

        for img_data in self.image_metadata:
            file_path = os.path.normpath(img_data["metadata"]["file_path"])
            if not os.path.isabs(file_path):
                file_path = os.path.abspath(file_path)

            self.paths.append(file_path)
            self.labels.append(self.label_to_idx[img_data["class_name"]])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        file_path = self.paths[idx]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        img = Image.open(file_path).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, idx

    def get_filename(self, idx):
        return os.path.basename(self.paths[idx])

class RSIDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.transform = transform
        self.paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = []
        self.image_metadata = []

        with open(json_path, "r", encoding="utf-8") as f:
            self.image_metadata = json.load(f)

        class_names = sorted(set(item["class_name"] for item in self.image_metadata))
        self.label_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.idx_to_label = class_names

        for img_data in self.image_metadata:
            file_path = os.path.normpath(img_data["metadata"]["file_path"])
            if not os.path.isabs(file_path):
                file_path = os.path.abspath(file_path)
            self.paths.append(file_path)
            self.labels.append(self.label_to_idx[img_data["class_name"]])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        file_path = self.paths[idx]

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        img = Image.open(file_path).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, idx

    def get_filename(self, idx):
        return os.path.basename(self.paths[idx])

def classify_images(model, dataset):
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0)
    results = []

    with torch.no_grad():
        for images, labels, indices in tqdm(dataloader):
            images = images.cuda()
            labels = labels.cuda()

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ zeroshot_weights
            pred = torch.argmax(logits, dim=1)

            for i, idx in enumerate(indices):
                results.append({
                    "image_id": dataset.get_filename(idx),
                    "correct_class": dataset.idx_to_label[labels[i].item()],
                    "predicted_class": dataset.idx_to_label[pred[i].item()]
                })

    return results


def create_zeroshot_classifier(dataset):
    classnames = dataset.idx_to_label
    templates = ["a photo of a {}.", "a satellite image of a {}."]

    zeroshot_weights = []
    with torch.no_grad():
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]
            print("current texts",texts)
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

    return torch.stack(zeroshot_weights, dim=1).cuda()


def find_inconsistent_tags(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        inconsistent_records = []
        error_stats = defaultdict(int)

        for item in data:
            expected_id = label_to_id.get(item['class_name'], -1)
            generated_tag = item.get('generated_tags', '')
            if isinstance(generated_tag, int):
                generated_id = generated_tag
            else:
                cleaned_tag = generated_tag.strip()
                if cleaned_tag.startswith('Tag:'):
                    cleaned_tag = cleaned_tag[4:].strip()
                generated_id = int(cleaned_tag) if cleaned_tag.isdigit() else -1
            if expected_id != generated_id:
                inconsistent_records.append(item)
                error_type = f"Expected {expected_id}({id_to_label.get(expected_id, 'unknown')}), got {generated_id}"
                error_stats[error_type] += 1
        if inconsistent_records:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(inconsistent_records, f, indent=2)
        else:
            print("All tags are consistent, no inconsistent records found")

        return inconsistent_records, error_stats

    except Exception as e:
        print(f"An error occurred during the processing: {str(e)}")
        return None, None


def update_tag_results(result_path, knowledge_base_path, output_path):
    global label_to_id
    with open(result_path, 'r') as f:
        results = json.load(f)
    with open(knowledge_base_path, 'r') as f:
        knowledge_base = json.load(f)

    kb_dict = {item["image_id"]: item for item in knowledge_base}
    updated_count = 0
    for result in results:
        image_id = result["image_id"]
        correct_class = result["correct_class"]
        predicted_class = result["predicted_class"]
        if correct_class == predicted_class and image_id in kb_dict:
            if dataset == 'UCM':
                label_to_id = ucm_label_to_id
            elif dataset == 'RSI':
                label_to_id = rsi_label_to_id
            elif dataset == 'RS':
                label_to_id = rs2800_label_to_id
            class_id = label_to_id.get(predicted_class, -1)
            if class_id != -1:
                kb_dict[image_id]["generated_tags"] = str(class_id)
                updated_count += 1
    with open(output_path, 'w') as f:
        json.dump(list(kb_dict.values()), f, indent=4)





if __name__ == "__main__":
    first_tag_path = "../rag/rs/knowledge_base_clip_rs.json"
    model, preprocess = clip.load("ViT-L/14")
    model.eval()
    retag_results = "../result/few-shot/retag_results_rs.json"
    dataset_name = 'UCM'
    dataset = UCMercedLandDataset( first_tag_path, transform=preprocess)
    if dataset_name == 'UCM':
        dataset = UCMercedLandDataset(first_tag_path, transform=preprocess)
    elif dataset_name == 'RSI':
        dataset = RSIDataset(first_tag_path, transform=preprocess)
    elif dataset_name == 'RS':
        dataset = RS2800Dataset( first_tag_path, transform=preprocess)
    print("Creating Zero-Shot Classifier...")
    zeroshot_weights = create_zeroshot_classifier(dataset)
    print("Done.\n")
    print("Classifying Images...")
    results = classify_images(model, dataset)
    with open(retag_results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    final_tag_path = "../result/few-shot/final_tags_result_rs.json"
    update_tag_results(retag_results, first_tag_path, final_tag_path)
