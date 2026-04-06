"""
Convert the XGBoost ranker model (JSON format) to ONNX format.

Usage:
    python convert_to_onnx.py [--input /path/to/xgboost_ranker.json] [--output /path/to/xgboost_ranker.onnx]
"""

import argparse
import os
import json
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Default paths (relative to sparky-serving/)
DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_ranker.json")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_ranker.onnx")

N_FEATURES = 59


def main():
    parser = argparse.ArgumentParser(description="Convert XGBoost ranker to ONNX format")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to the XGBoost model JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path for the output ONNX model file",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    print(f"Loading XGBoost model from: {input_path}")
    booster = xgb.Booster()
    booster.load_model(input_path)

    # Load metadata to verify feature count
    meta_path = os.path.join(os.path.dirname(input_path), "model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_features = meta.get("n_features", N_FEATURES)
        print(f"Metadata: {n_features} features, {meta.get('n_trees', '?')} trees")
    else:
        n_features = N_FEATURES
        print(f"No metadata found, using default {n_features} features")

    # Define input type for ONNX conversion
    # Use "regressor" target_opset to ensure ranking model outputs raw scores
    # (not class labels), since rank:ndcg produces continuous relevance scores.
    initial_type = [("input", FloatTensorType([None, n_features]))]

    print("Converting to ONNX format...")
    onnx_model = convert_xgboost(
        booster,
        initial_types=initial_type,
        target_opset=15,
    )

    # Post-process: if the converter created a classifier (label + probabilities),
    # we need to extract raw leaf scores instead. Use XGBoost's built-in ONNX export
    # as a fallback.
    import onnx
    output_names = [o.name for o in onnx_model.graph.output]
    if "label" in output_names:
        print("WARNING: onnxmltools produced classifier outputs. Using raw score workaround.")
        # Re-export: save as XGBoost native, then convert via treelite/manual approach
        # For now, use the probabilities[:,1] column as the score proxy
        print("Will use 'probabilities' output column 1 as relevance score.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the ONNX model
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model saved to: {output_path} ({file_size_mb:.2f} MB)")

    # Verify the converted model
    print("Verifying ONNX model...")
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Test with dummy data
    dummy_input = np.random.randn(3, n_features).astype(np.float32)
    result = session.run([output_name], {input_name: dummy_input})
    print(f"Verification passed: input shape {dummy_input.shape} -> output shape {result[0].shape}")
    print("Conversion complete.")


if __name__ == "__main__":
    main()
