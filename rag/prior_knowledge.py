import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from util.text_description_set import ucm_candidate_descriptions, rsi_cb256_descriptions, rs_image_descriptions

#Dataset Path
dataset_path = "../query_data/RS_images_2800"
knowledge_base_path = "../rag/rs/knowledge_base_clip_rs.json"


knowledge_base = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

dataset_name = 'UCM'
candidate_descriptions = ucm_candidate_descriptions

for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        for image_name in sorted(os.listdir(class_path)):
            if not image_name.lower().endswith('.jpg'):
                continue
            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path).convert("RGB")
            if dataset_name == 'UCM':
                candidate_descriptions = ucm_candidate_descriptions
            elif dataset_name == 'RSI':
                candidate_descriptions = rsi_cb256_descriptions
            elif dataset_name == 'RS':
                candidate_descriptions = rs_image_descriptions
            inputs = processor(
                text=candidate_descriptions,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            best_description_idx = probs.argmax()
            best_description = candidate_descriptions[best_description_idx]
            width, height = image.size
            format = "JPEG"
            entry = {
                "image_id": image_name,
                "class_name": class_name,
                "description": best_description,
                "metadata": {
                    "width": width,
                    "height": height,
                    "format": format,
                    "file_path": image_path
                }
            }
            knowledge_base.append(entry)


with open(knowledge_base_path, "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, ensure_ascii=False, indent=4)
