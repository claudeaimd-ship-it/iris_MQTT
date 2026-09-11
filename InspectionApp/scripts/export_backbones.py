"""
export_backbones.py — Export MobileNetV2 intermediate blocks as resolution-independent ONNX files.

Generates teacher_mobilenetv2_backbone_b{N}.onnx for blocks b3–b17 into
data/models/backbones/.  All files use dynamic spatial axes so the same
backbone works at any resize resolution (no re-export needed when
resize_to_training_resolution changes).

Requirements (run on PC or Jetson — NOT required on Raspberry Pi):
    pip install torch torchvision onnx

Usage:
    cd /path/to/InspectionApp
    python3 scripts/export_backbones.py             # export all b3–b17
    python3 scripts/export_backbones.py --blocks 3 7 9   # export specific blocks
    python3 scripts/export_backbones.py --force     # overwrite existing files

The backbones are the same architecture regardless of the target resolution.
Only one export run is needed per machine.  Copy the resulting .onnx files
to data/models/backbones/ on every device that runs calibration.
"""
import argparse
import os
import sys


# ── Block → return node mapping for MobileNetV2 features ─────────────────────
# MobileNetV2 `features` is a Sequential of 19 sub-modules (0–18):
#   0 = first Conv
#   1–18 = InvertedResidual blocks (called "layers" in PaDiM literature)
# We export the output of `{block}.conv.0` which is the depthwise conv output
# of the chosen InvertedResidual block.  Blocks b1–b17 map to indices 1–17.
# Blocks b16–b17 correspond to the last two InvertedResidual blocks before
# the classifier; they produce high-level abstract feature maps.

def _build_model(block: int):
    """Return a CPU, eval-mode ONNX-exportable backbone for the given block."""
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torchvision.models.feature_extraction import create_feature_extractor

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    backbone = models.mobilenet_v2(weights=weights).features

    return_nodes = {f"{block}.conv.0": "features"}

    class BackboneNCHW(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = create_feature_extractor(backbone, return_nodes=return_nodes)
            for p in self.net.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.net(x)["features"]

    # Wrap with NHWC permute so the ONNX graph accepts (1, H, W, 3) input —
    # consistent with the NumPy pipeline in CalibrationService and PaDiMInferenceAdapter.
    class BackboneNHWC(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = BackboneNCHW()

        def forward(self, x):
            # x: (N, H, W, 3) — NHWC
            x = x.permute(0, 3, 1, 2)     # → NCHW
            out = self.core(x)              # → (N, C, H', W')
            return out.permute(0, 2, 3, 1) # → (N, H', W', C)

    return BackboneNHWC().cpu().eval()


def export_block(block: int, output_dir: str, force: bool = False) -> None:
    import torch

    out_path = os.path.join(output_dir, f"teacher_mobilenetv2_backbone_b{block}.onnx")

    if os.path.exists(out_path) and not force:
        print(f"[SKIP] b{block} already exists (use --force to overwrite): {out_path}")
        return

    print(f"[EXPORT] Building backbone for block b{block}…")
    model = _build_model(block)

    # Use a representative dummy input — spatial size only affects the output
    # feature map size, not the model weights.  Dynamic axes let the exported
    # model accept any (H, W) at runtime.
    dummy = torch.randn(1, 224, 224, 3)
    with torch.no_grad():
        out = model(dummy)
    print(f"         Input dummy (1,224,224,3) → output {tuple(out.shape)}")

    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=13,
        input_names=["input_rgb"],
        output_names=["features"],
        dynamic_axes={
            "input_rgb": {0: "batch", 1: "height", 2: "width"},
            "features":  {0: "batch", 1: "feat_h", 2: "feat_w"},
        },
    )
    print(f"[OK]     Saved: {out_path}")


def verify_block(block: int, output_dir: str, test_shape: tuple = (525, 525)) -> None:
    """Quick sanity check: run the exported ONNX at test_shape and print output shape."""
    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError:
        print("[SKIP] onnxruntime not installed — skipping verification.")
        return

    path = os.path.join(output_dir, f"teacher_mobilenetv2_backbone_b{block}.onnx")
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found — export may have failed.")
        return

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, test_shape[0], test_shape[1], 3).astype(np.float32)
    out = sess.run(None, {"input_rgb": dummy})[0]
    print(f"[VERIFY] b{block} @ {test_shape}: input {dummy.shape} → output {out.shape}  ✓")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--blocks", type=int, nargs="+", default=list(range(3, 18)),
        metavar="N", help="Block numbers to export (default: 3..17).",
    )
    parser.add_argument(
        "--output", type=str, default="data/models/backbones",
        metavar="DIR", help="Output directory (default: data/models/backbones).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .onnx files.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run a quick shape-check with onnxruntime after each export.",
    )
    parser.add_argument(
        "--verify-shape", type=int, nargs=2, default=[525, 525],
        metavar=("H", "W"), help="Spatial shape to use for verification (default: 525 525).",
    )
    args = parser.parse_args()

    # Resolve output dir relative to the repo root (where this script is run from).
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    try:
        import torch          # noqa: F401
        import torchvision    # noqa: F401
    except ImportError:
        print(
            "[ERROR] torch and torchvision are required to export backbones.\n"
            "        pip install torch torchvision\n"
            "        (only needed on the machine that exports — not on the Raspberry Pi)"
        )
        sys.exit(1)

    print(f"Output directory: {output_dir}")
    print(f"Blocks to export: {args.blocks}")
    print()

    for block in args.blocks:
        export_block(block, output_dir, force=args.force)
        if args.verify:
            verify_block(block, output_dir, test_shape=tuple(args.verify_shape))

    print()
    print("[DONE] All backbones exported.")
    print(f"       Copy {output_dir}/*.onnx to data/models/backbones/ on every device.")


if __name__ == "__main__":
    main()
