"""
Split the pre-trained transformers into four components:
(a) embedding layer;
(b) freeze layers;
(c) fine-tuned layers;
(d) classification head;

Preserve (a) and (b) as they will be shared across different tasks.
"""
import torch
import argparse
from transformers import (
    BertModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from constants import MODEL_NAME, FREEZE_LAYER_COUNT  


def export_shared_architecture(model_name: str, freeze_layer_count: int, output_dir: str) -> None:
    # 1. Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Export tokenizer
    tokenizer.save_pretrained(output_dir) # TODO: check with the team do we share tokenizer

    # Export embedding layers
    torch.save(model.bert.embeddings.state_dict(), f'{output_dir}/embeddings_layer.pt')

     # Export frozen layers
    frozen_layers = model.bert.encoder.layer[:freeze_layer_count]
    frozen_layers_state_dicts = [layer.state_dict() for layer in frozen_layers]
    torch.save(frozen_layers_state_dicts, f'{output_dir}/frozen_layers.pt')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze_layer_count", type=int, default=FREEZE_LAYER_COUNT)
    parser.add_argument("--model_name", type=str, default= MODEL_NAME)
    parser.add_argument("--out_folder", type=str, default= "etc/ml_models/feature_extractor")
    args = parser.parse_args()

    export_shared_architecture(args.model_name, args.freeze_layer_count, args.out_folder)