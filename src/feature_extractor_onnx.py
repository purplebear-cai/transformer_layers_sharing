"""
Split the pre-trained transformers into four components:
(a) embedding layer;
(b) freeze layers;
(c) fine-tuned layers;
(d) classification head;

Preserve (a) and (b) as they will be shared across different tasks.
"""
import torch
import numpy as np
from torch import nn, Tensor
import onnxruntime as ort
from onnxruntime import InferenceSession
from transformers import (
    BertModel,
    AutoTokenizer,
    BertTokenizerFast,
    AutoModelForSequenceClassification,
)
from transformers.models.bert.modeling_bert import BertEmbeddings
from optimum.onnxruntime import ORTQuantizer
from onnxruntime.quantization import quantize_dynamic, QuantType
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from constants import MAX_LEN, HIDDEN_SIZE


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
    

class FeatureExtractorOnnx(nn.Module):
    def __init__(self, pretrained_model_name, feature_folder, quantized=False):
        super(FeatureExtractorOnnx, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        if not quantized:
            self.embedding_session = self.load_onnx_model(f"{feature_folder}/embeddings_layer.onnx")
            self.frozen_layer_session = self.load_onnx_model(f"{feature_folder}/frozen_layers.onnx")
        else:
            self.embedding_session = self.load_onnx_model(f"{feature_folder}/embeddings_layer_quantized.onnx")
            self.frozen_layer_session = self.load_onnx_model(f"{feature_folder}/frozen_layers_quantized.onnx")

    @staticmethod
    def load_onnx_model(onnx_file: str) -> InferenceSession:
        """
        Load ONNX model as InferenceSession.
        """
        session = ort.InferenceSession(onnx_file)
        return session

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # Run embedding session
        embedding_session_input_name = self.embedding_session.get_inputs()[0].name
        embedding_session_input_add_name = self.embedding_session.get_inputs()[1].name
        embedding_session_inputs = {
            embedding_session_input_name: input_ids.numpy(), 
            embedding_session_input_add_name: token_type_ids.numpy()
            }
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
    
    def extract(self, texts):
        """
        Extract contextual embeddings given texts. The high-level pipeline looks like below:
        texts --> embedding session --> frozen_layer session --> contextual embeddings
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            contextual_embeddings = self(input_ids, token_type_ids, attention_mask)

        return contextual_embeddings

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

    # remove_input(onnx_file + ".ori", "onnx::Add_1", onnx_file)
    

def remove_input(model_path, input_name, output_model_path):
        import onnx
        from onnx import helper
        model = onnx.load(model_path)

        # Find and remove the input from the graph input if it exists
        input_to_remove = None
        for input_tensor in model.graph.input:
            if input_tensor.name == input_name:
                input_to_remove = input_tensor
                break

        if input_to_remove:
            model.graph.input.remove(input_to_remove)
        else:
            print(f"Input '{input_name}' not found in the model.")

        # Remove the input from the initializers if it exists
        for i, init in enumerate(model.graph.initializer):
            if init.name == input_name:
                model.graph.initializer.pop(i)
                break

        # Save the modified model
        onnx.save(model, output_model_path)
        

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


def quantize_onnx_model(onnx_model_path, quantized_model_path) -> None:
    """
    Convert onnx model to quantized onnx model.
    """
    # dynamic_quantizer = ORTQuantizer.from_pretrained(onnx_model_path)
    # dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    # dynamic_quantizer.quantize(save_dir=quantized_model_path, quantization_config=dqconfig)
    quantize_dynamic(
        onnx_model_path, 
        quantized_model_path, 
        weight_type=QuantType.QInt8,
    )

def quantize_shared_architecture(onnx_folder) -> None:
    """
    Quantize the embedding layers and frozen layers for faster inference and smaller model size.
    """
    quantize_onnx_model(f"{onnx_folder}/embeddings_layer.onnx", f"{onnx_folder}/embeddings_layer_quantized.onnx")
    quantize_onnx_model(f"{onnx_folder}/frozen_layers.onnx", f"{onnx_folder}/frozen_layers_quantized.onnx")