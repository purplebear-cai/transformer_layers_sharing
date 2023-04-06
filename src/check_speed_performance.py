"""
Assuming that there are 1000 sentences, for each sentence, we will run four transformer-based classifiers.

Setting 1:
    For each sentence, we run four pipelines, each pipelien contains embedding --> encoder --> pooler --> classifier;

Setting 2:
    For each sentence, we run one encoding pipeline: embedding --> encoder (freeze layer),
                       and run four classification pipeline: encoder --> (fined-tuned layer) --> pooler --> classifier;
"""
import time
import torch
import argparse
import pandas as pd
from torch import nn
from sentence_classifier import FineTunedBertClassifier
from sentence_classifier import load_modules, prepare_input
from transformers import (
    pipeline,
    Trainer,
    BertModel, 
    BertConfig,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
)

NUM_CLASSIFIERS = 4

def load_data():
    limits = 500
    folder = "/Users/qcai/Workspace/Projects/transformer_layers_sharing/"
    val_path = folder + "etc/ml_models/pubmed/inputs/validation.csv"

    val_df = pd.read_csv(val_path).dropna()
    val_df = val_df.head(limits)

    val_texts, val_labels = list(val_df["text"]), list(val_df["label"])
    return val_texts, val_labels

def get_speed_for_setting_1(checkpoint_dir, val_texts, val_labels):
    print("Loading model from checkpoints ...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)   
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    print("Start inference ...")
    start_time = time.time()
    for _ in range(NUM_CLASSIFIERS):
        val_preds = []
        for val_text in val_texts:
            pred_label = classifier(val_text)[0]["label"]
            val_preds.append(pred_label)
    duration = time.time() - start_time   
    print(f"Setting 1: running time = {duration}s")
    # print(f"accuracy: {accuracy_score(val_labels, val_preds)}")


def get_speed_for_setting_2(model_name, feature_extractor_dir, 
                            fine_tuned_dir, freeze_layer_count, 
                            val_texts, val_labels):
    print("Loading each independent modules")
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

    freeze_layers = model.encoder.layer[0:freeze_layer_count]
    fine_tuned_layers = model.encoder.layer[freeze_layer_count:]


    print("Start inference ...")
    start_time = time.time()
    val_preds = []
    for val_text in val_texts:
        # Prepare input
        input_ids, attention_mask, token_type_ids = prepare_input(tokenizer, val_text)

        # Run the embedding layers and frozen layers
        with torch.no_grad():
            input_embeddings = model.embeddings(input_ids, token_type_ids)
            hidden_states = input_embeddings 
            for layer in freeze_layers:
                hidden_states = layer(hidden_states, attention_mask)[0]
            
        # Run the fine-tuned layers
        for _ in range(NUM_CLASSIFIERS):
            with torch.no_grad():
                for layer in fine_tuned_layers:
                    hidden_states = layer(hidden_states, attention_mask)[0]

                sequence_output = hidden_states
                pooled_output = model.pooler(sequence_output)

                # Pass the output of the fine-tuned layers through the classification head
                logits = classification_head(pooled_output)
                
                # Decode the logits to obtain the predicted class
                predicted_id= torch.argmax(logits, dim=-1).item()
                predicted_label = config.id2label[predicted_id]
                val_preds.append(predicted_label)
    duration = time.time() - start_time   
    print(f"Setting 2: running time = {duration}s")
    # print(f"accuracy: {accuracy_score(val_labels, val_preds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="pubmed")
    parser.add_argument("--freeze_layer_count", type=int, default=2)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--keep-checkpoint", default=True, action="store_true")
    parser.add_argument("--model_name", type=str, default= "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument("--feature_extractor_dir", type=str, default="etc/ml_models/feature_extractor")
    parser.add_argument("--fine_tuned_dir", type=str, default= "etc/ml_models/pubmed/results/1.0")
    parser.add_argument("--checkpoint_dir", type=str, default="etc/ml_models/pubmed/results/1.0/checkpoint-192")

    args = parser.parse_args()

    val_texts, val_labels = load_data()
    get_speed_for_setting_1(args.checkpoint_dir, val_texts, val_labels)
    get_speed_for_setting_2(args.model_name, 
                            args.feature_extractor_dir, 
                            args.fine_tuned_dir, 
                            args.freeze_layer_count, 
                            val_texts, 
                            val_labels)
    print("done.")