import os
import math
import torch
import argparse
import numpy as np
import pandas as pd

from torch import nn
from datasets import DatasetDict, Dataset
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
)
from transformers import (
    Trainer,
    BertModel, 
    BertConfig,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)

MAX_LEN = 512


class FineTunedBertClassifier(nn.Module):
    def __init__(self, config, tokenizer, embeddings, bert_model, classification_head):
        super(FineTunedBertClassifier, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        self.encoder = bert_model.encoder
        self.pooler = bert_model.pooler
        self.classification_head = classification_head

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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def get_pubmed_dataset(tokenizer: AutoTokenizer):
    """
    Read pubmed data to build a classifier.

    :param: AutoTokenizer, HuggingFace tokenizer
    """
    limits = 1000
    folder = "/Users/qcai/Workspace/Projects/transformer_layers_sharing/"
    train_path = folder + "etc/ml_models/pubmed/inputs/train.csv"
    val_path = folder + "etc/ml_models/pubmed/inputs/validation.csv"

    # load training set
    train_df = pd.read_csv(train_path).dropna()
    train_df = train_df.head(limits) if limits > -1 else train_df
    labels_set = set(train_df["label"])
    label2id = {l:i for i, l in enumerate(labels_set)}
    id2label = {i:l for l, i in label2id.items()}
    train_df["label_id"] = train_df.apply(lambda row: label2id[row["label"]], axis=1)

    # load validation set
    val_df = pd.read_csv(val_path).dropna()
    val_df = val_df.head(300) # TODO: remove the head
    val_df["label_id"] = val_df.apply(lambda row: label2id[row["label"]], axis=1)

    # build training dataset
    train_texts, train_labels = list(train_df["text"]), list(train_df["label_id"])
    train_data = pd.DataFrame({"text": train_texts, "label": train_labels})
    train_ds = Dataset.from_pandas(train_data)
    
    # build validation dataset
    val_texts, val_labels = list(val_df["text"]), list(val_df["label_id"])
    val_data = pd.DataFrame({"text": val_texts, "label": val_labels})
    val_ds = Dataset.from_pandas(val_data)

    # build the dataset for transformers, containing train and test
    ds = DatasetDict({"train": train_ds, "test": val_ds})
    ds = ds.map(lambda e: tokenizer(e["text"], padding=False, truncation=True), batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return ds, label2id


def save_pt(tokenizer, model_bert, model_classifier, freeze_layer_count, output_dir):
    # Export tokenizer
    tokenizer.save_pretrained(output_dir)

    # Export embedding layers
    torch.save(model_bert.embeddings.state_dict(), f'{output_dir}/embeddings_layer.pt')

    # Separate models for frozen layers and fine-tuned layers
    frozen_layers = model_bert.encoder.layer[:freeze_layer_count]
    fine_tuned_layers = model_bert.encoder.layer[freeze_layer_count:]

    # Export frozen layers
    frozen_layers_state_dicts = [layer.state_dict() for layer in frozen_layers]
    torch.save(frozen_layers_state_dicts, f'{output_dir}/frozen_layers.pt')

    # Export fine-tuned layers
    fine_tuned_layers_state_dicts = [layer.state_dict() for layer in fine_tuned_layers]
    torch.save(fine_tuned_layers_state_dicts, f'{output_dir}/fine_tuned_layers.pt')

    # Export classification head
    torch.save(model_classifier.state_dict(), f"{output_dir}/classification_head.pt")


def train(
    output_dir: str,
    dataset_name: str,
    freeze_layer_count: int = 0,
    model_name: str = None,
):
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

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained(model_name)    

    # Load dataset
    if dataset_name == "pubmed":
        ds, label2id = get_pubmed_dataset(tokenizer)
    else:
        raise ValueError("Unknown dataset name.")
    train_ds, val_ds = ds["train"], ds["test"]

    # Update and save config (this is task specific)
    model.config.label2id = label2id
    model.config.id2label = {i:l for l, i in label2id.items()}
    model.config.save_pretrained(output_dir)

    # Freeze layers if required
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

    # Define training arguments
    epoch_steps = len(train_ds) / args_dict["per_device_train_batch_size"]
    args_dict["warmup_steps"] = math.ceil(epoch_steps)  # 1 epoch
    args_dict["logging_steps"] = max(1, math.ceil(epoch_steps * 0.5))  # 0.5 epoch
    args_dict["save_steps"] = args_dict["logging_steps"]
    args_dict["load_best_model_at_end"] = True
    training_args = TrainingArguments(output_dir=str(output_dir), **args_dict)

    # Start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()

    # Export each 
    save_pt(tokenizer, model.bert, model.classifier, freeze_layer_count, output_dir)

    print("Training is successfully completed.")


def load_modules(model_name, feature_extractor_dir, fine_tuned_dir, freeze_layer_count):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(feature_extractor_dir)
        
    # Load pre-trained config and initial model architure
    config = BertConfig.from_pretrained(fine_tuned_dir)
    model = BertModel.from_pretrained(model_name, config=config)

    # Load embeddings
    saved_embeddings = torch.load(f"{feature_extractor_dir}/embeddings_layer.pt")
    model.embeddings.load_state_dict(saved_embeddings) # load embeddings
    
    # Load frozen layers
    saved_frozen_layers = torch.load(f'{feature_extractor_dir}/frozen_layers.pt')
    for i, layer_state_dict in enumerate(saved_frozen_layers):
        model.encoder.layer[i].load_state_dict(layer_state_dict)
    
    # Load fine-tuned layers
    saved_fine_tuned_layers = torch.load(f'{fine_tuned_dir}/fine_tuned_layers.pt')
    for i, layer_state_dict in enumerate(saved_fine_tuned_layers):
        model.encoder.layer[i + freeze_layer_count].load_state_dict(layer_state_dict)

    # Load classification head
    saved_classification_head = torch.load(f"{fine_tuned_dir}/classification_head.pt")
    classification_head = nn.Linear(config.hidden_size, 5) # TODO here 5 is the num_labels
    classification_head.load_state_dict(saved_classification_head)

    # Create the fine-tuned model 
    fine_tuned_model = FineTunedBertClassifier(
        config,
        tokenizer,
        model.embeddings, 
        model,
        classification_head, 
    )

    return fine_tuned_model

def prepare_input(tokenizer: AutoTokenizer, text: str):
    """
    Prepare inputs to fine-tuned bert.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    return inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]

def eval(fine_tuned_model, dataset_name):
    """
    Evaluate the model.
    """
    label2id = fine_tuned_model.config.label2id
    id2label = {id: label for label, id in label2id.items()}
    
    if dataset_name == "pubmed":
        limits = 100
        folder = "/Users/qcai/Workspace/Projects/transformer_layers_sharing/"
        val_path = folder + "etc/ml_models/pubmed/inputs/validation.csv"

        val_df = pd.read_csv(val_path).dropna()
        val_df = val_df.head(limits)

        val_texts, val_labels = list(val_df["text"]), list(val_df["label"])
    else:
        raise ValueError("Unknown dataset name.")
    
    val_preds = []
    for val_text, val_label in zip(val_texts, val_labels):
        # Prepare input
        input_ids, attention_mask, token_type_ids = prepare_input(fine_tuned_model.tokenizer, val_text)

        # Run inference
        with torch.no_grad():
            logits = fine_tuned_model(input_ids, attention_mask, token_type_ids)

        # Decode the logits to obtain the predicted class
        predicted_id= torch.argmax(logits, dim=-1).item()
        predicted_label = id2label[predicted_id]
        val_preds.append(predicted_label)

        print(f"expected={val_label}, predicted_id={predicted_id}, predicted_label={predicted_label}, text={val_text}")

    print(f"\n=============== Performance Report ===============")
    y_true = val_labels
    y_pred = val_preds
    print(f"\nAccuracy = {accuracy_score(y_true, y_pred)}")

    print("\nConfusion matrix = ")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="pubmed")
    parser.add_argument("--freeze_layer_count", type=int, default=2)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--keep-checkpoint", default=True, action="store_true")
    parser.add_argument("--model_name", type=str, default= "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument("--feature_extractor_dir", type=str, default="etc/ml_models/feature_extractor")
    parser.add_argument("--output_dir", type=str, default= "etc/ml_models/pubmed/results/1.0")

    args = parser.parse_args()
    print(f"** Train size: {args.train_size} **")
    print(f"** Freeze layers: {args.freeze_layer_count} **")

    # train(
    #     output_dir=args.output_dir,
    #     dataset_name=args.dataset_name,
    #     freeze_layer_count=args.freeze_layer_count,
    #     model_name=args.model_name,
    # )

    fine_tuned_model = load_modules(args.model_name, 
                                    args.feature_extractor_dir, 
                                    args.output_dir, 
                                    args.freeze_layer_count)
    eval(fine_tuned_model, args.dataset_name)
 