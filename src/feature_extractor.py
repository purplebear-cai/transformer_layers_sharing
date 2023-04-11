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
from torch import nn, Tensor
import onnxruntime as ort
from onnxruntime import InferenceSession
from transformers import (
    AutoModel,
    BertModel,
    AutoTokenizer,
    BertTokenizerFast,
    AutoModelForSequenceClassification,
)
from transformers.models.bert.modeling_bert import BertEmbeddings
from constants import MODEL_NAME, FREEZE_LAYER_COUNT, MAX_LEN, HIDDEN_SIZE

class FrozenBertModel(nn.Module):
    def __init__(self, bert, freeze_layer_count):
        super(FrozenBertModel, self).__init__()
        self.bert = bert
        self.freeze_layer_count = freeze_layer_count

    def forward(self, embeddings, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        hidden_states = embeddings
        for _, layer in enumerate(self.bert.encoder.layer[0:self.freeze_layer_count]):
            hidden_states = layer(hidden_states, extended_attention_mask)[0]
        return hidden_states
    

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model_name, feature_folder, num_frozen_layers):
        super(FeatureExtractor, self).__init__()
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


class FeatureExtractorOnnx(nn.Module):
    def __init__(self, feature_folder):
        super(FeatureExtractorOnnx, self).__init__()
        self.embedding_session = self.load_onnx_model(f"{feature_folder}/embeddings_layer.onnx")
        self.frozen_layer_session = self.load_onnx_model(f"{feature_folder}/frozen_layers.onnx")

    @staticmethod
    def load_onnx_model(onnx_file: str) -> InferenceSession:
        """
        Load ONNX model as InferenceSession.
        """
        session = ort.InferenceSession(onnx_file)
        return session

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # Run embedding session
        embedding_session_input_name = self.embedding_session.get_inputs()[0].name
        embedding_session_inputs = {embedding_session_input_name: input_ids.numpy()}
        embeddings_session_output_name = self.embedding_session.get_outputs()[0].name
        embeddings = self.embedding_session.run([embeddings_session_output_name], embedding_session_inputs)[0]

        # Run frozen_layer session
        frozen_session_input_name = self.frozen_layer_session.get_inputs()[0].name
        frozen_session_attention_mask_name = self.frozen_layer_session.get_inputs()[1].name
        frozen_session_inputs = {
            frozen_session_input_name: embeddings,
            frozen_session_attention_mask_name: attention_mask.numpy()
        }
        frozen_session_output_name = self.frozen_layer_session.get_outputs()[0].name
        frozen_outputs = self.frozen_layer_session.run([frozen_session_output_name], frozen_session_inputs)[0]
        return torch.from_numpy(frozen_outputs)
    

def export_shared_architecture_as_onnx(model_name: str, freeze_layer_count: int, output_dir: str) -> None:
    """
    Export embedding layer and frozen layers as ONNX format.
    """
    # 1. Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Export tokenizer
    tokenizer.save_pretrained(output_dir)

    # Export embedding layers
    save_embedding_as_onnx(model.bert.embeddings, tokenizer, f'{output_dir}/embeddings_layer.onnx')

    # Export frozen layers
    save_frozen_layers_as_onnx(model.bert, freeze_layer_count, f'{output_dir}/frozen_layers.onnx')


def export_shared_architecture(model_name: str, freeze_layer_count: int, output_dir: str) -> None:
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


def save_embedding_as_onnx(model: BertEmbeddings, tokenizer: BertTokenizerFast, onnx_file: str) -> None:
    """
    Save embedding layers as ONNX format.
    """
    model.eval()
    encoded_input = tokenizer.encode_plus("This is a dummy example.", 
                                        max_length=MAX_LEN, 
                                        truncation=True, 
                                        padding='max_length', 
                                        return_tensors='pt')
    dummy_input = encoded_input['input_ids']
    dynamic_axes = {"input_ids": {0: "batch_size"}, "embeddings": {0: "batch_size"}} # to allow dynamic batching during inference
    torch.onnx.export(model, 
                      dummy_input, 
                      onnx_file, 
                      input_names=['input_ids'], 
                      output_names=['embeddings'], 
                      dynamic_axes=dynamic_axes, 
                      opset_version=12)


def save_frozen_layers_as_onnx(bert_model: BertModel, freeze_layer_count: int, onnx_file: str) -> None:
    """
    Save frozen layers as ONNX format.
    """
    frozen_bert_model = FrozenBertModel(bert_model, freeze_layer_count)
    frozen_bert_model.eval()

    dummy_input = torch.zeros([1, MAX_LEN, HIDDEN_SIZE])
    dummy_attention_mask = torch.ones(1, MAX_LEN).long()
    input_names = ["embeddings", "attention_mask"]
    dynamic_axes = {"embeddings": {0: "batch_size"}, "attention_mask": {0: "batch_size"}, "frozen_states": {0: "batch_size"}}
    torch.onnx.export(frozen_bert_model, 
                      (dummy_input, dummy_attention_mask), 
                      onnx_file, 
                      input_names=input_names, 
                      output_names=['frozen_states'], 
                      dynamic_axes=dynamic_axes, 
                      opset_version=12)


def extract_contextual_embeddings(texts: list, 
                                  feature_extractor_model: FeatureExtractor, 
                                  tokenizer: BertTokenizerFast) -> Tensor:

    """
    Extract contextual embeddings given texts. The high-level pipeline looks like below:
    texts --> pre-trained embeddings [batch_size, MAX_LEN, HIDDEN_SIZE] --> frozen layers --> contextual embeddings
    """
    inputs = tokenizer(texts, return_tensors="pt", max_length=MAX_LEN, padding='max_length', truncation=True)
    with torch.no_grad():
        contextual_embeddings = feature_extractor_model(inputs["input_ids"], inputs["attention_mask"])
    return contextual_embeddings

def extract_contextual_embeddings_onnx(texts: list, 
                                       feature_extractor_onnx_model: FeatureExtractorOnnx, 
                                       tokenizer: BertTokenizerFast):
    """
    Extract contextual embeddings given texts. The high-level pipeline looks like below:
    texts --> embedding session --> frozen_layer session --> contextual embeddings
    """
    inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        contextual_embeddings = feature_extractor_onnx_model(input_ids, attention_mask)

    return contextual_embeddings


def extract_features_with_pt(model_name: str, 
                             feature_folder: str, 
                             freeze_layer_count: int, 
                             texts: list):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the custom feature extractor model
    feature_extractor_model = FeatureExtractor(model_name, feature_folder, freeze_layer_count)

    # Extract features
    contextual_embeddings = extract_contextual_embeddings(texts, feature_extractor_model, tokenizer)
    return contextual_embeddings


def extract_features_with_onnx(model_name: str, 
                               feature_folder: str, 
                               texts: list):
    # Load inference session, tokenizer, feature extractor
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    feature_extractor_onnx_model = FeatureExtractorOnnx(feature_folder)

    # Extract features
    contextual_embeddings = extract_contextual_embeddings_onnx(texts, feature_extractor_onnx_model, tokenizer)
    return contextual_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze_layer_count", type=int, default=FREEZE_LAYER_COUNT)
    parser.add_argument("--model_name", type=str, default= MODEL_NAME)
    parser.add_argument("--feature_folder", type=str, default= "etc/ml_models/feature_extractor")
    args = parser.parse_args()
    
    # EXPORT SHARED ARCHITECTURE
    export_shared_architecture(args.model_name, args.freeze_layer_count, f"{args.feature_folder}/pt")
    export_shared_architecture_as_onnx(args.model_name, args.freeze_layer_count, f"{args.feature_folder}/onnx")

    # INFERENCE WITH SHARED ARCHITECTUR
    # Goal: Inference with .pt formats and .onnx formats should have the same contextual embedding outputs
    texts = ["This is an example", "Another text to process"]
    contextual_embeddings_pt = extract_features_with_pt(args.model_name, f"{args.feature_folder}/pt", args.freeze_layer_count, texts)
    print(f"\ncontextual_embeddings_pt = \n{contextual_embeddings_pt}")

    contextual_embeddings_onnx = extract_features_with_onnx(args.model_name, f"{args.feature_folder}/onnx", texts)
    print(f"\ncontextual_embeddings_onnx = \n{contextual_embeddings_onnx}")