import json
import argparse
import math
import os
import torch
from torch import nn
import pandas as pd
from pathlib import Path
import shutil
from typing import Optional
import uuid

import datasets
from datasets import DatasetDict, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoConfig,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, BertConfig



class FineTunedBertClassifier(nn.Module):
    def __init__(self, bert_model, embeddings, classification_head, config):
        super(FineTunedBertClassifier, self).__init__()
        self.embeddings = embeddings
        self.encoder = bert_model.encoder
        self.pooler = bert_model.pooler
        self.classification_head = classification_head
        self.config = config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Use the separated embeddings layer
        input_embeddings = self.embeddings(input_ids, token_type_ids)
        
        # Use the BERT model's encoder and pooler
        encoder_outputs = self.encoder(input_embeddings, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output)

        # Use the classification head
        logits = self.classification_head(pooled_output)
        return logits




def get_random_seed():
    return int.from_bytes(os.urandom(4), "big")


DATASET_MAP = {"sst2": ("glue", "sst2"), "cola": ("glue", "cola"), "imdb": ("imdb",)}
SPLIT_DIR = Path("split")


def get_split_path(dataset_name: str, train_size: int):
    if dataset_name not in DATASET_MAP:
        raise ValueError(f"unknown dataset: {dataset_name}")

    dataset_tuple = DATASET_MAP[dataset_name]
    return SPLIT_DIR / f"{dataset_tuple[0]}-{dataset_tuple[1]}-{train_size}.npy"


def get_dataset(tokenizer, dataset_name: str, split: str, split_path: Optional[Path] = None):
    ds = datasets.load_dataset(*DATASET_MAP[dataset_name], split=split)
    ds = ds.shuffle(seed=42)

    if split_path is not None:
        # split_path is a npy file containing indexes of samples to keep
        print(f"Using split file {split_path}")
        split_ids = set(np.load(split_path).tolist())
        ds = ds.filter(lambda idx: idx in split_ids, input_columns="idx")

    ds = ds.map(lambda e: tokenizer(e["text"], padding=False, truncation=True), batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def get_qq_dataset(tokenizer):
    limits = 1000
    folder = "/Users/qcai/Workspace/Projects/transformer_projects/"
    train_path = folder + "etc/ml_models/pubmed/inputs/train.csv"
    val_path = folder + "etc/ml_models/pubmed/inputs/validation.csv"

    train_df = pd.read_csv(train_path).dropna()
    train_df = train_df.head(limits) if limits > -1 else train_df
    labels_set = set(train_df["label"])
    label2id = {l:i for i, l in enumerate(labels_set)}
    id2label = {i:l for l, i in label2id.items()}
    train_df["label_id"] = train_df.apply(lambda row: label2id[row["label"]], axis=1)

    val_df = pd.read_csv(val_path).dropna()
    val_df = val_df.head(300)
    val_df["label_id"] = val_df.apply(lambda row: label2id[row["label"]], axis=1)

    train_texts, train_labels = list(train_df["text"]), list(train_df["label_id"])
    val_texts, val_labels = list(val_df["text"]), list(val_df["label_id"])

    train_data = pd.DataFrame({"text": train_texts, "label": train_labels})
    val_data = pd.DataFrame({"text": val_texts, "label": val_labels})

    train_ds = Dataset.from_pandas(train_data)
    val_ds = Dataset.from_pandas(val_data)
    ds = DatasetDict({"train": train_ds, "test": val_ds})

    # ds = ds.train_test_split(test_size=0.2)
    # dataset_dict = DatasetDict({"train": datasets["train"], "test": datasets["test"]})
    # ds = ds.shuffle(seed=42)

    ds = ds.map(lambda e: tokenizer(e["text"], padding=False, truncation=True), batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds, label2id


def save_pt(tokenizer, model_bert, model_classifier, freeze_layer_count, folder):
    tokenizer.save_pretrained(folder)

    torch.save(model_bert.embeddings.state_dict(), f'{folder}/embeddings_layer.pt')

    # Separate models for frozen layers and fine-tuned layers
    frozen_layers = model_bert.encoder.layer[:freeze_layer_count]
    fine_tuned_layers = model_bert.encoder.layer[freeze_layer_count:]

    # Save the first two frozen layers
    frozen_layers_state_dicts = [layer.state_dict() for layer in frozen_layers]
    torch.save(frozen_layers_state_dicts, f'{folder}/frozen_layers.pt')

    # Export fine-tuned layers as ONNX
    fine_tuned_layers_state_dicts = [layer.state_dict() for layer in fine_tuned_layers]
    torch.save(fine_tuned_layers_state_dicts, f'{folder}/fine_tuned_layers.pt')

    # Export classification head as ONNX
    from transformers import BertForSequenceClassification

    # classification_model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    classification_head = model_classifier
    torch.save(classification_head.state_dict(), f"{folder}/classification_head.pt")



def train(
    output_dir: str,
    dataset_name: str,
    train_size: Optional[int] = None,
    freeze_layer_count: int = 0,
    model_name: str = None,
):
    split_path = get_split_path(dataset_name, train_size) if train_size is not None else None
    args_dict = {
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 5e-5,
        "num_train_epochs": 10,
        "logging_first_step": True,
        "save_total_limit": 1,
        "fp16": False,
        "dataloader_num_workers": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        # we need to generate a random seed manually, as otherwise
        # the same constant random seed is used during training for each run
        "seed": get_random_seed(),
    }

    config = BertConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=5)

    if freeze_layer_count:
        # We freeze here the embeddings of the model
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

        if freeze_layer_count != -1:
            # if freeze_layer_count == -1, we only freeze the embedding layer
            # otherwise we freeze the first `freeze_layer_count` encoder layers
            for layer in model.bert.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds, label2id = get_qq_dataset(tokenizer)
    train_ds, val_ds = ds["train"], ds["test"]


    model.config.label2id = label2id
    model.config.id2label = {i:l for l, i in label2id.items()}
    model.config.save_pretrained(output_dir)

    epoch_steps = len(train_ds) / args_dict["per_device_train_batch_size"]
    args_dict["warmup_steps"] = math.ceil(epoch_steps)  # 1 epoch
    args_dict["logging_steps"] = max(1, math.ceil(epoch_steps * 0.5))  # 0.5 epoch
    args_dict["save_steps"] = args_dict["logging_steps"]
    args_dict["load_best_model_at_end"] = True
    # args_dict["run_name"] = output_dir.name

    training_args = TrainingArguments(output_dir=str(output_dir), **args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    save_pt(tokenizer, model.bert, model.classifier, freeze_layer_count, output_dir)

    print("Finish training")


def load_pt(model_name, folder):
    # # Load label2id
    # with open(folder + "/label2id.json") as file:
    #     label2id = json.load(file)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(folder)

    # Load ONNX models
    saved_embeddings = torch.load(f"{folder}/embeddings_layer.pt")
    saved_frozen_layers = torch.load(f'{folder}/frozen_layers.pt')
    saved_fine_tuned_layers = torch.load(f'{folder}/fine_tuned_layers.pt')
    saved_classification_head = torch.load(f"{folder}/classification_head.pt")

    # Initialize the Microsoft BERT model
    config = BertConfig.from_pretrained(folder)  # Update this with the actual name of the pretrained model
    # config.label2id = label2id
    model = BertModel.from_pretrained(model_name, config=config)

    # Load the embeddings, frozen layers, and fine-tuned layers
    model.embeddings.load_state_dict(saved_embeddings)

    # Replace the model's encoder layers with the loaded layers
    for i, layer_state_dict in enumerate(saved_frozen_layers):
        model.encoder.layer[i].load_state_dict(layer_state_dict)

    for i, layer_state_dict in enumerate(saved_fine_tuned_layers):
        model.encoder.layer[i + 2].load_state_dict(layer_state_dict)

    # Load the classification head
    classification_head = nn.Linear(config.hidden_size, 5) # TODO here 5 is the num_labels
    classification_head.load_state_dict(saved_classification_head)

    # Create the fine-tuned model
    fine_tuned_model = FineTunedBertClassifier(model, model.embeddings, classification_head, config)

    return fine_tuned_model, tokenizer

def prepare_input(tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]

def eval(fine_tuned_model, tokenizer):
    label2id = fine_tuned_model.config.label2id
    id2label = {id: label for label, id in label2id.items()}
    limits = 100
    folder = "/Users/qcai/Workspace/Projects/transformer_projects/"
    val_path = folder + "etc/ml_models/pubmed/inputs/validation.csv"

    val_df = pd.read_csv(val_path).dropna()
    val_df = val_df.head(100)

    val_texts, val_labels = list(val_df["text"]), list(val_df["label"])
    val_preds = []
    for val_text, val_label in zip(val_texts, val_labels):
        input_ids, attention_mask, token_type_ids = prepare_input(tokenizer, val_text)

        # Run the model
        with torch.no_grad():
            logits = fine_tuned_model(input_ids, attention_mask, token_type_ids)

        # Decode the logits to obtain the predicted class
        predicted_class = torch.argmax(logits, dim=-1).item()
        predicted_labels = id2label[predicted_class]
        val_preds.append(predicted_labels)

        print(f"expected={val_label}, actual={predicted_class}, text={val_text}")

    print(f"\n=============== Performance Report ===============")
    y_true = val_labels
    y_pred = val_preds
    print("\n==============")
    print(accuracy_score(y_true, y_pred))

    print("\n==============")
    print(confusion_matrix(y_true, y_pred))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="imdb")
    parser.add_argument("--freeze_layer_count", type=int, default=2)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--keep-checkpoint", default=True, action="store_true")
    parser.add_argument("--model_name", type=str, default= "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument("--folder", type=str, default= "/Users/qcai/Downloads/last_try")

    args = parser.parse_args()
    OUTPUT_DIR = Path(args.folder)
    print(f"** Train size: {args.train_size} **")
    print(f"** Freeze layers: {args.freeze_layer_count} **")
    output_dir = OUTPUT_DIR / str(uuid.uuid4())
    train(
        output_dir=args.folder,
        dataset_name=args.dataset_name,
        train_size=args.train_size,
        freeze_layer_count=args.freeze_layer_count,
        model_name=args.model_name,
    )
    # # if not args.keep_checkpoint:
    # #     shutil.rmtree(output_dir)

    fine_tuned_model, tokenizer = load_pt(args.model_name, args.folder)

    eval(fine_tuned_model, tokenizer)


    