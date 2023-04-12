import time
import argparse
from constants import MODEL_NAME, FREEZE_LAYER_COUNT

from feature_extractor_pt import FeatureExtractorPt, export_shared_architecture_as_pt
from feature_extractor_onnx import FeatureExtractorOnnx, export_shared_architecture_as_onnx, quantize_shared_architecture


def extract_features_with_pt(model_name: str, 
                             feature_folder: str, 
                             freeze_layer_count: int, 
                             texts: list):
    # Create the custom feature extractor model
    feature_extractor_model = FeatureExtractorPt(model_name, feature_folder, freeze_layer_count)

    # Extract features
    contextual_embeddings = feature_extractor_model.extract(texts)
    return contextual_embeddings


def extract_features_with_onnx(model_name: str, 
                               feature_folder: str, 
                               texts: list):
    # Load inference session, tokenizer, feature extractor
    feature_extractor_onnx_model = FeatureExtractorOnnx(model_name, feature_folder)

    # Extract features
    contextual_embeddings = feature_extractor_onnx_model.extract(texts)
    return contextual_embeddings


def extract_features_with_quantized_onnx(model_name: str, 
                                         feature_folder: str, 
                                         texts: list):
    feature_extractor_onnx_model = FeatureExtractorOnnx(model_name, feature_folder, quantized=True)
    contextual_embeddings = feature_extractor_onnx_model.extract(texts)
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
    # quantize_shared_architecture(f"{args.feature_folder}/onnx")

    # INFERENCE WITH SHARED ARCHITECTUR
    # Goal: Inference with .pt formats and .onnx formats should have the same contextual embedding outputs
    texts = ["This is an example", "Another text to process"]
    start_time = time.time()
    contextual_embeddings_pt = extract_features_with_pt(args.model_name, f"{args.feature_folder}/pt", args.freeze_layer_count, texts)
    print(f"\nExtracting with pt: {time.time()-start_time}s")
    print(f"contextual_embeddings_pt = \n{contextual_embeddings_pt}")

    start_time = time.time()
    contextual_embeddings_onnx = extract_features_with_onnx(args.model_name, f"{args.feature_folder}/onnx", texts)
    print(f"\nExtracting with onnx: {time.time()-start_time}s")
    print(f"contextual_embeddings_onnx = \n{contextual_embeddings_onnx}")

    start_time = time.time()
    contextual_embeddings_onnx = extract_features_with_quantized_onnx(args.model_name, f"{args.feature_folder}/onnx", texts)
    print(f"\nExtracting with quantized onnx: {time.time()-start_time}s")
    print(f"contextual_embeddings_quantized_onnx = \n{contextual_embeddings_onnx}")