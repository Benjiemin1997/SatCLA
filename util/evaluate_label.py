import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 定义label到id的映射字典
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

rs_label_to_id = {
    "farmland": 0,
    "forest": 1,
    "grassland": 2,
    "industry": 3,
    "parkinglot": 4,
    "resident": 5,
    "river": 6,
}

dataset = 'UCM'
label_to_id = []
if dataset == 'UCM':
    label_to_id = ucm_label_to_id
elif dataset == 'RSI':
    label_to_id = rsi_label_to_id
elif dataset == 'RS':
    label_to_id = rs_label_to_id

id_to_label = {v: k for k, v in label_to_id.items()}


def evaluate_performance(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    true_labels = []
    pred_labels = []

    for item in data:
        true_class = item["class_name"]
        true_id = label_to_id.get(true_class, -1)
        try:
            pred_id = int(item["generated_tags"])
            pred_class = id_to_label.get(pred_id, "unknown")
        except (ValueError, TypeError):
            continue
        if true_id != -1 and pred_id != -1:
            true_labels.append(true_id)
            pred_labels.append(pred_id)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Total Samples": len(true_labels),
        "Number of Classes": len(label_to_id)
    }

if __name__ == "__main__":
    json_path = "../result/ucm/final_result/final_tags_result_granite_rs.json"
    metrics = evaluate_performance(json_path)
    with open("../result/ucm/final_result/evaluate_granite.json", 'w') as f:
        json.dump(metrics, f, indent=4)