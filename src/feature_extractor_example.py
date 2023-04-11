import torch
import argparse
from torch import Tensor
from transformers import AutoTokenizer, BertTokenizerFast
from constants import MAX_LEN, MODEL_NAME, FREEZE_LAYER_COUNT

from feature_extractor_pt import FeatureExtractorPt, export_shared_architecture_as_pt
from feature_extractor_onnx import FeatureExtractorOnnx, export_shared_architecture_as_onnx

def extract_contextual_embeddings_pt(texts: list, 
                                     feature_extractor_model: FeatureExtractorPt, 
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
    feature_extractor_model = FeatureExtractorPt(model_name, feature_folder, freeze_layer_count)

    # Extract features
    contextual_embeddings = extract_contextual_embeddings_pt(texts, feature_extractor_model, tokenizer)
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
    
    # # EXPORT SHARED ARCHITECTURE
    # export_shared_architecture_as_pt(args.model_name, args.freeze_layer_count, f"{args.feature_folder}/pt")
    # export_shared_architecture_as_onnx(args.model_name, args.freeze_layer_count, f"{args.feature_folder}/onnx")

    # INFERENCE WITH SHARED ARCHITECTUR
    # Goal: Inference with .pt formats and .onnx formats should have the same contextual embedding outputs
    texts = ["This is an example", "Another text to process"]
    contextual_embeddings_pt = extract_features_with_pt(args.model_name, f"{args.feature_folder}/pt", args.freeze_layer_count, texts)
    print(f"\ncontextual_embeddings_pt = \n{contextual_embeddings_pt}")

    contextual_embeddings_onnx = extract_features_with_onnx(args.model_name, f"{args.feature_folder}/onnx", texts)
    print(f"\ncontextual_embeddings_onnx = \n{contextual_embeddings_onnx}")