"""Export the one-step generator and the RAEv2 decoder to ONNX for CPU inference, verify against PyTorch.

    uv run --extra demo python scripts/export_onnx.py --student runs/student_gan_fixed/student.safetensors
    uv run --extra demo python scripts/export_onnx.py --student ... --int8     # + weight-only int8 decoder

Writes onnx/generator_1step.onnx, onnx/decoder.onnx (and decoder_int8.onnx) plus onnx/export_report.json with
the max deviation from PyTorch and the median CPU latency of each graph.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from humangen.eval.generate import load_generator
from humangen.rae import RAE
from humangen.rae.autoencoder import grid_to_tokens


class OneStepGenerator(nn.Module):
    """noise (B,1024,16,16), attrs (B,K) int64 -> normalised latent x̂ predicted at t=1."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, noise: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        t = torch.ones(noise.shape[0], dtype=noise.dtype, device=noise.device)
        return self.model(noise, t, attrs)[0]


class LatentDecoder(nn.Module):
    """normalised latent (B,1024,16,16) -> image (B,3,256,256) in [0,1]."""

    def __init__(self, rae: RAE):
        super().__init__()
        self.normalizer, self.decoder = rae.normalizer, rae.decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(grid_to_tokens(self.normalizer.denormalize(z))).clamp(0, 1)


def export(module: nn.Module, args: tuple, names: list[str], path: Path) -> None:
    batch = torch.export.Dim("batch", min=1, max=64)
    shapes = tuple(
        {0: batch} for _ in names
    )  # positional: forward() arg names differ from the ONNX input names
    prog = torch.onnx.export(
        module,
        args,
        dynamo=True,
        input_names=names,
        output_names=["out"],
        dynamic_shapes=shapes,
        optimize=True,
        verbose=False,
    )
    n_bytes = sum(t.numel() * t.element_size() for t in [*module.parameters(), *module.buffers()])
    prog.save(str(path), external_data=n_bytes > 1.8e9)  # protobuf's 2 GB limit (the decoder is ~1.7 GB)


def ort_session(path: Path, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def latency_ms(fn, reps: int) -> float:
    fn()  # warm-up
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(1000 * (time.perf_counter() - t0))
    return statistics.median(times)


def file_mb(path: Path) -> float:
    """Graph plus any external-data sidecar (large initialisers are written to <name>.onnx.data)."""
    return round(sum(p.stat().st_size for p in path.parent.glob(path.name + "*")) / 1e6, 1)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="one-step student (or any generator) safetensors")
    ap.add_argument("--out", default="onnx")
    ap.add_argument(
        "--int8", action="store_true", help="also write a weight-only int8 decoder (3.5x smaller)"
    )
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = all cores)")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cpu = torch.device("cpu")

    model, attr_names, _ = load_generator(a.student, cpu)
    gen = OneStepGenerator(model.float()).eval()
    dec = LatentDecoder(RAE(load_encoder=False)).eval()
    k = model.cfg.num_attributes
    noise = torch.randn(2, 1024, 16, 16)
    attrs = torch.zeros(2, k, dtype=torch.long)
    attrs[1, : min(k, 3)] = 2  # exercise the attribute embedding with a mix of states

    report: dict = {"student": str(a.student), "attr_names": attr_names, "torch": torch.__version__}
    with torch.no_grad():
        z_ref = gen(noise, attrs)
        img_ref = dec(z_ref)
    for name, module, args, names in (
        ("generator_1step", gen, (noise, attrs), ["noise", "attrs"]),
        ("decoder", dec, (z_ref,), ["latent"]),
    ):
        path = out / f"{name}.onnx"
        t0 = time.time()
        with torch.no_grad():
            export(module, args, names, path)
        report[name] = {"export_s": round(time.time() - t0, 1), "mb": file_mb(path)}
        print(f"[onnx] {path} ({report[name]['mb']} MB) in {report[name]['export_s']} s", flush=True)

    import onnxruntime as ort

    report["onnxruntime"] = ort.__version__
    g_sess, d_sess = (
        ort_session(out / "generator_1step.onnx", a.threads),
        ort_session(out / "decoder.onnx", a.threads),
    )
    feeds = {"noise": noise.numpy(), "attrs": attrs.numpy()}
    z_ort = g_sess.run(None, feeds)[0]
    img_ort = d_sess.run(None, {"latent": z_ref.numpy()})[0]
    report["generator_1step"]["max_abs_diff"] = float(np.abs(z_ort - z_ref.numpy()).max())
    report["decoder"]["max_abs_diff"] = float(np.abs(img_ort - img_ref.numpy()).max())
    report["decoder"]["psnr_vs_torch_db"] = round(psnr(img_ort, img_ref.numpy()), 1)

    one = {"noise": feeds["noise"][:1], "attrs": feeds["attrs"][:1]}
    z1 = z_ref.numpy()[:1]
    report["generator_1step"]["cpu_ms_per_face"] = round(latency_ms(lambda: g_sess.run(None, one), a.reps), 1)
    report["decoder"]["cpu_ms_per_face"] = round(
        latency_ms(lambda: d_sess.run(None, {"latent": z1}), a.reps), 1
    )
    with torch.no_grad():
        report["torch_cpu_ms_per_face"] = {
            "generator_1step": round(latency_ms(lambda: gen(noise[:1], attrs[:1]), a.reps), 1),
            "decoder": round(latency_ms(lambda: dec(z_ref[:1]), a.reps), 1),
        }

    if a.int8:
        # Weight-only, block-wise int8 (MatMulNBits). Dynamic int8 also quantises activations per tensor, and the
        # ViT-XL decoder's activation outliers wreck it (measured 9-11 dB PSNR vs fp32); int4 blocks give 24.6 dB.
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        import onnx

        q_path = out / "decoder_int8.onnx"
        qz = MatMulNBitsQuantizer(
            onnx.load(str(out / "decoder.onnx")), bits=8, block_size=32, is_symmetric=True, accuracy_level=4
        )
        qz.process()
        onnx.save_model(qz.model.model, str(q_path), save_as_external_data=False)
        q_sess = ort_session(q_path, a.threads)
        img_q = q_sess.run(None, {"latent": z_ref.numpy()})[0]
        report["decoder_int8"] = {
            "method": "weight-only int8, block 32 (MatMulNBits)",
            "mb": file_mb(q_path),
            "psnr_vs_fp32_db": round(psnr(img_q, img_ref.numpy()), 1),
            "cpu_ms_per_face": round(latency_ms(lambda: q_sess.run(None, {"latent": z1}), a.reps), 1),
        }

    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "attr_names"}, indent=2))


if __name__ == "__main__":
    main()
