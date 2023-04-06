import json
import pandas as pd
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from constants import MODEL_NAME
from transformers import (
    AutoTokenizer,
)

def build_datasetdict(tokenizer, train_df, val_df):
    """
    Build HuggingFace DatasetDict based on the given training and validation dataframes.
    """
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
    return ds

def load_pubmed_dataset(tokenizer: AutoTokenizer):
    """
    Read pubmed data to build a classifier.

    :param: AutoTokenizer, HuggingFace tokenizer
    """
    train_df, val_df, label2id = load_pubmed_text()
    ds = build_datasetdict(tokenizer, train_df, val_df)
    return ds, label2id


def load_pubmed_text():
    limits = 1000
    folder = "/Users/qcai/Workspace/Projects/transformer_layers_sharing/"
    train_path = folder + "etc/ml_models/pubmed/inputs/train.csv"
    val_path = folder + "etc/ml_models/pubmed/inputs/validation.csv"

    # load training set
    train_df = pd.read_csv(train_path).dropna()
    train_df = train_df.head(limits) if limits > -1 else train_df
    
    # load validation set
    val_df = pd.read_csv(val_path).dropna()
    val_df = val_df.head(300) # TODO: remove the head

    labels_set = set(train_df["label"])
    label2id = {l:i for i, l in enumerate(labels_set)}
    train_df["label_id"] = train_df.apply(lambda row: label2id[row["label"]], axis=1)
    val_df["label_id"] = val_df.apply(lambda row: label2id[row["label"]], axis=1)
    
    return train_df, val_df, label2id
    



def load_narrative_dataset(tokenizer):
    """
    Load narrative data to build a narrative classifier.
    """
    train_df, val_df, label2id = load_narrative_text()
    ds = build_datasetdict(tokenizer, train_df, val_df)
    return ds, label2id

def load_narrative_text():
    """
    Load narrative text.
    """
    in_path = "/Users/caiq/Workspace/olive/transformer_layers_sharing/etc/ml_models/narrative/inputs/narrative_nvp_binary_v2_8.vlearn.json"
    with open(in_path) as in_file:
        data = json.load(in_file)
    texts, labels = [], []
    for item in data:
        text = item["text"]
        start_idx = text.index("SENT_START")
        end_idx = text.index("SENT_END")
        if start_idx != -1:
            text = text[start_idx+len("SENT_START"):]
        if end_idx != -1:
            text = text[0:end_idx]
        texts.append(text)

        label = item["categories"][0]
        labels.append(label)
    df = pd.DataFrame({"text": texts, "label": labels})
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    labels_set = set(train_df["label"])
    label2id = {l:i for i, l in enumerate(labels_set)}

    train_df["label_id"] = train_df.apply(lambda row: label2id[row["label"]], axis=1)
    val_df["label_id"] = val_df.apply(lambda row: label2id[row["label"]], axis=1)

    return train_df, val_df, label2id


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    load_narrative_dataset(tokenizer)