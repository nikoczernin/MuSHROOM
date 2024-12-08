import json
import torch
from transformers import AdamW
from torch.nn import CrossEntropyLoss

from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification
from torch.utils.data import DataLoader, Dataset
from baseline_utils import get_data_for_training

import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


# create a class for the dataset
class HallucinationDataset(Dataset):
    def __init__(self, features, labels, tokenizer, max_length):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        labels = self.labels[idx]

        # Tokenize input text
        encoded = self.tokenizer(
            feature,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # label_ids starts as a list of -100 values, matching the length of the tokenized input sequence.
        label_ids = [-100] * len(encoded["input_ids"][0])  # Ignore non-response tokens
        # the self.tokenizer.sep_token_id identifies the [SEP] token, which separates the query from the response
        # response_start points to the position in the tokenized sequence immediately after the first [SEP],
        # marking the start of the response
        response_start = encoded["input_ids"][0].tolist().index(self.tokenizer.sep_token_id) + 1
        # labels is the binary label list for the response tokens (e.g., [0, 0, 1, 0, 1]).
        # This line replaces the corresponding portion of label_ids (starting from response_start) with
        # the actual labels for the response tokens.
        label_ids[response_start:response_start + len(labels)] = labels

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label_ids, dtype=torch.long),
        }






def train_model(model, dataloader, args, tokenizer_name="mbert_token_classifier"):
    model.to(args.device)
    print("Model moved to device!")
    # Step 5: Training loop
    model.train()
    print("Training started!")
    for epoch in range(args.num_epochs):  # Number of epochs
        total_loss = 0
        for batch in dataloader:
            # Move data to device
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["label"].to(args.device)
            # Forward pass
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            # Backward pass
            args.optimizer.zero_grad()
            loss.backward()
            args.optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
    print("Training complete!")
    # Step 6: Save the model
    model.save_pretrained(args.model_name)
    tokenizer.save_pretrained("mbert_token_classifier")
    print("Model and tokenizer saved!")


def inference(model, inputs):
    # inputs: input_ids of a batch, which is the yield of iterating the trainloader
    # for batch in trainloader:
    #     inference(model, batch["input_ids"])
    # returns:
    # TokenClassifierOutput with attrs: loss, logits, grad_fn, hidden_states, attentions
    # the logits should be the probabilities of all classes for each observation
    # and therefore have shape (num_obs_in_batch, size_feature_space, num_class_labels)
    # model.eval()   # set model to inference mode
    predictions = model(inputs)
    print(predictions)
    return predictions


features, labels = get_data_for_training('../data/preprocessed/sample_preprocessed.json')
print("Data is prepared!")

# Define constants
TOKENIZER_MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128  # Max token length for mBERT


print("Loading tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
# model = BertForSequenceClassification.from_pretrained(TOKENIZER_MODEL_NAME, num_labels=2)
model = BertForTokenClassification.from_pretrained(TOKENIZER_MODEL_NAME, num_labels=2)
print("Tokenizer and model loaded!")




# Create datasets and dataloaders
train = HallucinationDataset(features, labels, tokenizer, max_length=MAX_LENGTH)
test = HallucinationDataset(features, labels, tokenizer, max_length=MAX_LENGTH)
trainloader = DataLoader(train, batch_size=8, shuffle=True)
testloader = DataLoader(test, batch_size=8, shuffle=True)
print("Datasets and dataloaders created!")

# Training setup
# Helper class for neural network hyper-parameters
class Args:
    pass
ARGS = Args()
ARGS.use_cuda = AdamW(model.parameters(), lr=2e-5)
# Define optimizer and loss function
ARGS.loss_fn = CrossEntropyLoss()
# Set device
ARGS.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {ARGS.device}")
ARGS.model_name = "mbert_token_classifier"
ARGS.num_epochs = 3
### TRAINING
# train_model(model, trainloader, args=ARGS)

### TESTING
print("we testing now")
model.eval() # is this necessary?
for batch in testloader:
    classifier_output = inference(model, batch["input_ids"])
    yhat = classifier_output["logits"]


