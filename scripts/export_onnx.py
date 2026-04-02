#!/usr/bin/env python3
"""
Export nemotron-speech-streaming-en-0.6b to ONNX for sherpa-onnx.
Adapted from sherpa-onnx/scripts/nemo/nemotron-speech-streaming-en-0.6b/export_onnx.py
Modified to use local .nemo checkpoint and run in controlled stages.

Output defaults to ~/.local/share/telemuze/models/nemotron-speech-streaming-en-0.6b/
"""
import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Dict

import torch


DEFAULT_OUTPUT_DIR = (
    Path.home() / ".local" / "share" / "telemuze" / "models" / "nemotron-speech-streaming-en-0.6b"
)


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place."""
    import onnx

    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    import os
    filepath = os.path.abspath(filename)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    onnx.save(
        model,
        filepath,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=basename + ".data",
    )
    del model
    gc.collect()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Export nemotron-speech-streaming-en-0.6b to ONNX")
    parser.add_argument(
        "--nemo-path",
        default="/home/scott/code/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo",
        help="Path to .nemo checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=None,
        help="Streaming chunk size in ms (e.g. 80, 160, 560, 1120). "
             "Selects from the model's supported sizes. Default: largest available.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    nemo_path = args.nemo_path
    print(f"Step 1: Loading model from {nemo_path}...")
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.ASRModel.restore_from(nemo_path)
    print("Model loaded.")

    # Save tokens
    with open(output_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
    print("Saved tokens.txt")

    asr_model.eval()

    # Extract streaming config — select chunk size from model's supported sizes.
    # Each frame is 10ms, so chunk_size in frames * 10 = latency in ms.
    assert asr_model.encoder.streaming_cfg is not None
    cfg = asr_model.encoder.streaming_cfg

    chunk_sizes = cfg.chunk_size if isinstance(cfg.chunk_size, list) else [cfg.chunk_size]
    cache_sizes = cfg.pre_encode_cache_size if isinstance(cfg.pre_encode_cache_size, list) else [cfg.pre_encode_cache_size]
    available_ms = [cs * 10 for cs in chunk_sizes]
    print(f"Available chunk sizes: {available_ms} ms (frames: {chunk_sizes})")

    if args.chunk_ms is not None:
        target_frames = args.chunk_ms // 10
        if target_frames not in chunk_sizes:
            print(f"Error: --chunk-ms {args.chunk_ms} not in supported sizes: {available_ms} ms")
            sys.exit(1)
        idx = chunk_sizes.index(target_frames)
    else:
        # Default: largest available (last entry)
        idx = len(chunk_sizes) - 1

    chunk_size = chunk_sizes[idx]
    pre_encode_cache_size = cache_sizes[idx]
    window_size = chunk_size + pre_encode_cache_size

    print(f"Selected: chunk_size={chunk_size} frames ({chunk_size * 10}ms), "
          f"pre_encode_cache_size={pre_encode_cache_size}, window_size={window_size}")

    chunk_shift = chunk_size

    cache_last_channel_dim1 = len(asr_model.encoder.layers)
    cache_last_channel_dim2 = asr_model.encoder.streaming_cfg.last_channel_cache_size
    cache_last_channel_dim3 = asr_model.encoder.d_model

    cache_last_time_dim1 = len(asr_model.encoder.layers)
    cache_last_time_dim2 = asr_model.encoder.d_model
    cache_last_time_dim3 = asr_model.encoder.conv_context_size[0]

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "window_size": window_size,
        "chunk_shift": chunk_shift,
        "normalize_type": normalize_type,
        "cache_last_channel_dim1": cache_last_channel_dim1,
        "cache_last_channel_dim2": cache_last_channel_dim2,
        "cache_last_channel_dim3": cache_last_channel_dim3,
        "cache_last_time_dim1": cache_last_time_dim1,
        "cache_last_time_dim2": cache_last_time_dim2,
        "cache_last_time_dim3": cache_last_time_dim3,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "feat_dim": 128,
        "model_type": "EncDecHybridRNNTCTCBPEModel",
        "version": "1",
        "model_author": "NeMo",
        "url": "https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b",
        "comment": "Only the transducer branch is exported",
    }
    print("Meta data:", meta_data)

    # Step 2: Export each component one at a time
    asr_model.set_export_config({"cache_support": True})

    print("\nStep 2a: Exporting encoder.onnx...")
    sys.stdout.flush()
    asr_model.encoder.export(str(output_dir / "encoder.onnx"))
    gc.collect()
    print("Encoder exported.")

    print("\nStep 2b: Exporting decoder.onnx...")
    sys.stdout.flush()
    asr_model.decoder.export(str(output_dir / "decoder.onnx"))
    gc.collect()
    print("Decoder exported.")

    print("\nStep 2c: Exporting joiner.onnx...")
    sys.stdout.flush()
    asr_model.joint.export(str(output_dir / "joiner.onnx"))
    gc.collect()
    print("Joiner exported.")

    # Free the NeMo model before metadata/quantization
    del asr_model
    gc.collect()
    print("\nModel released from memory.")

    # Step 3: Add metadata to encoder
    print("\nStep 3: Adding metadata to encoder.onnx...")
    sys.stdout.flush()
    add_meta_data(str(output_dir / "encoder.onnx"), meta_data)
    print("Metadata added.")

    # Step 4: Quantize one at a time
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for m in ["encoder", "decoder", "joiner"]:
        print(f"\nStep 4: Quantizing {m}.onnx -> {m}.int8.onnx...")
        sys.stdout.flush()
        quantize_dynamic(
            model_input=str(output_dir / f"{m}.onnx"),
            model_output=str(output_dir / f"{m}.int8.onnx"),
            weight_type=QuantType.QUInt8,
        )
        gc.collect()
        print(f"{m}.int8.onnx created.")

    print("\nDone! All files exported successfully.")


if __name__ == "__main__":
    main()
