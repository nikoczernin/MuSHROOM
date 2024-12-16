import json

# Load JSONL data
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Load your files
predictions = load_jsonl("data/val/gpt/mushroom.ar-val.v2.spans.jsonl")
ground_truth = load_jsonl("data/val/mushroom.ar-val.v2.jsonl")

# Initialize counters
correct = 0
total_ground_truth = 0

# Compare predictions and ground truth
for pred, truth in zip(predictions, ground_truth):
    pred_labels = pred.get('indices', [])
    truth_labels = truth.get('hard_labels', [])
    print(pred_labels, truth_labels)
    # Ensure pred_labels is not None
    if pred_labels is None:
        pred_labels = []
    
    # Calculate correctly predicted true hard labels
    correct += len([label for label in truth_labels if label in pred_labels])
    
    # Count total ground truth labels
    total_ground_truth += len(truth_labels)

# Calculate recall
recall = correct / total_ground_truth if total_ground_truth > 0 else 0

# Print results
print(f"Recall: {recall:.2f}")
