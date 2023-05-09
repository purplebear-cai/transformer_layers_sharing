import torch
import random
import pandas as pd

from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix
from feature_extractor_pt import FeatureExtractorPt

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.classifier = torch.nn.Linear(model.config.hidden_size, 2)


class CombinedEmbeddingsClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CombinedEmbeddingsClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input embeddings
        return self.fc(x)


def load_data():
    rest_data = load_csv("/Users/caiq/Downloads/Rest_4773.csv", "Rest", 
                         "synonyms: bed rest, bedrest, resting; label: procedure")
    smoking_data = load_csv("/Users/caiq/Downloads/Smoking_116.csv", "Smoking", 
                            "synonyms: smoke, smoker, smoking, tabacoo; label: disease")
    return pd.concat([rest_data, smoking_data])

def load_csv(data_path, concept, concept_details):
    data_df = pd.read_csv(data_path)
    input_data = []
    for _, row in data_df.iterrows():
        text = row["PerceptValue"]
        review_value = row["Review"]
        if review_value == "Correct":
            label = "correct"
        if review_value == "Error":
            label = "error"
        input_data.append({
            "text": text, 
            "concept": concept,
            "concept_details": concept_details,
            "label": label,
        })
    input_data_df = pd.DataFrame(input_data)
    return input_data_df



def accuracy(outputs, labels):
    probabilities = torch.sigmoid(outputs)
    predictions = torch.argmax(probabilities, dim=-1)
    correct = (predictions == torch.argmax(labels, dim=-1)).sum().item()
    return correct / labels.size(0)


def train_model(model, train_dataloader, eval_dataloader, num_epochs=3, learning_rate=0.00001):
    
    # Initialize the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_accuracy += acc

        # Evaluate the model on the evaluation data
        model.eval()
        eval_loss = 0.0
        eval_accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in eval_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = accuracy(outputs, labels)
                eval_loss += loss.item()
                eval_accuracy += acc

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_dataloader):.4f}, Accuracy: {running_accuracy / len(train_dataloader):.4f}, Eval Loss: {eval_loss / len(eval_dataloader):.4f}, Eval Accuracy: {eval_accuracy / len(eval_dataloader):.4f}")


def inference(model, feature_extractor_model, label2id, 
              texts, concept_details, expected_labels):
    # Extract the embeddings for the text and concept_details
    text_embedding = feature_extractor_model.extract(texts)
    concept_details_embedding = feature_extractor_model.extract(concept_details)

    # Concatenate the embeddings
    concatenated_embedding = torch.cat((text_embedding, concept_details_embedding), dim=-1)

    # Get the prediction
    with torch.no_grad():
        outputs = model(concatenated_embedding)
        probabilities = torch.sigmoid(outputs)
        predictions = torch.argmax(probabilities, dim=-1)
    
    id2label = {i:l for l, i in label2id.items()}
    predicted_labels = [id2label[lid] for lid in predictions.tolist()]
    accuracy = accuracy_score(expected_labels, predicted_labels)
    cm = confusion_matrix(expected_labels, predicted_labels)
    print(f"\n=== confusion matrix: \n{cm}")
    print(f"\n=== accuracy: {accuracy}")
    return predictions


def load_models(input_dim, output_dim=2):
    # Load the trained model
    loaded_classifier = CombinedEmbeddingsClassifier(input_dim, output_dim)
    loaded_classifier.load_state_dict(torch.load(model_path))
    loaded_classifier.eval()
    return loaded_classifier


def convert_df_to_dataloader(data_df, label2id, batch_size, shuffle):
    texts = list(data_df["text"])
    concept_details = list(data_df["concept_details"])
    label_id_list = [label2id[label] for label in list(data_df["label"])]
    labels = torch.tensor([[1, 0] if lid == 0 else [0, 1] for lid in label_id_list], dtype=torch.float)

    train_text_embeddings = feature_extractor_model.extract(texts)
    train_concept_embeddings = feature_extractor_model.extract(concept_details)
    concatenated_embeddings = torch.cat((train_text_embeddings, train_concept_embeddings), dim=-1)

    train_dataset = TensorDataset(concatenated_embeddings, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    input_dim = train_text_embeddings.shape[1] * train_text_embeddings.shape[2] * 2  # Flatten the concatenated embeddings

    return train_dataloader, input_dim

# Load contextual embedding
feature_folder = "etc/ml_models/feature_extractor"
freeze_layer_count = 6
feature_extractor_model = FeatureExtractorPt(MODEL_NAME, f"{feature_folder}/pt", freeze_layer_count)
model_path = "etc/ml_models/pair_classifier/classifier.pth"
input_dim = 786432
output_dim = 2  # binary classification
label2id = {"error": 0, "correct": 1}

# Load data
print("Loading data ...")
input_data = load_data()
train_data, test_data = train_test_split(input_data, test_size = 0.2, stratify=input_data.label)

# # Start training
# print("Start training ...")
# train_dataloader, input_dim = convert_df_to_dataloader(train_data, label2id, batch_size=32, shuffle=True)
# eval_dataloader, input_dim = convert_df_to_dataloader(test_data, label2id, batch_size=32, shuffle=False)
# classifier = CombinedEmbeddingsClassifier(input_dim, output_dim)
# train_model(classifier, train_dataloader, eval_dataloader, num_epochs=30)
# torch.save(classifier.state_dict(), model_path)

# Start testing
print("Start evaluating ...")
loaded_classifier = load_models(input_dim)
test_texts = list(test_data["text"])
test_concept_details = list(test_data["concept_details"])
test_labels = list(test_data["label"])
inference(loaded_classifier, feature_extractor_model, label2id, 
          test_texts, test_concept_details, test_labels)

print("Done.")