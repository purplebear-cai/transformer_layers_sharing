"""
Split the pre-trained transformers into four components:
(a) embedding layer;
(b) freeze layers;
(c) fine-tuned layers;
(d) classification head;

Preserve (a) and (b) as they will be shared across different tasks.
"""
import torch
from torch import nn, Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertTokenizerFast,
    AutoModelForSequenceClassification,
)
from constants import MAX_LEN
    

class FeatureExtractorPt(nn.Module):
    def __init__(self, pretrained_model_name, feature_folder, num_frozen_layers):
        super(FeatureExtractorPt, self).__init__()
        pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
        self.embeddings = pretrained_model.embeddings
        self.transformer_layers = nn.ModuleList(pretrained_model.encoder.layer[:num_frozen_layers])
        self.init_states(feature_folder)

    def init_states(self, feature_folder):
        # Load the saved embeddings
        embeddings_state_dict = torch.load(f"{feature_folder}/embeddings_layer.pt")
        self.embeddings.load_state_dict(embeddings_state_dict)

        # Load the frozen layers from .pt files
        frozen_layers_state_dicts = torch.load(f"{feature_folder}/frozen_layers.pt")
        for i, layer_state_dict in enumerate(frozen_layers_state_dicts):
            self.transformer_layers[i].load_state_dict(layer_state_dict, strict=False)
        
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        x = self.embeddings(input_ids)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attention_mask)[0]
        return x


def export_shared_architecture_as_pt(model_name: str, freeze_layer_count: int, output_dir: str) -> None:
    """
    Export embedding layers and frozen layers as PT format.
    """
    # 1. Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Export tokenizer
    tokenizer.save_pretrained(output_dir)

    # Export embedding layers
    torch.save(model.bert.embeddings.state_dict(), f'{output_dir}/embeddings_layer.pt')
    
    # Export frozen layers
    frozen_layers = model.bert.encoder.layer[:freeze_layer_count]
    frozen_layers_state_dicts = [layer.state_dict() for layer in frozen_layers]
    torch.save(frozen_layers_state_dicts, f'{output_dir}/frozen_layers.pt')
