"""Export the models the website runs on each visitor's own device: ONNX graphs for ONNX Runtime Web (WebGPU on any
graphics chip, NVIDIA, AMD, Intel or Apple; WebAssembly on the processor when there is no WebGPU), compressed to
download over the web and cut into parts under Cloudflare's 25 MiB file limit.

    uv run --extra demo python scripts/site/export_web_models.py

Compression is weight-only (activations stay float32), measured against full precision on the same inputs; the numbers
go into the manifest and the page shows them:
  - one-step generator: 8-bit weights. 4-bit broke it (19 dB).
  - decoder: 8-bit in its first and last 4 blocks (and the in/out projections), 4-bit in the middle 20. All 8-bit was
    474 MB at 55 dB, all 4-bit 266 MB at 30 dB; the ends carry most of the error.
  - teacher (25-step "Best quality"): 8-bit weights. Same person and composition, fine texture differs (30-31 dB
    after 25 steps).
  - TrustMark watermark encoder / decoder (Adobe, MIT): float16 storage, cast back to float32 when loaded, and
    convolution inputs padded to a multiple of 4 channels (WebGPU miscomputes other counts; see pad_conv_channels).
Intermediate graphs stay in onnx/web/; the parts and manifest go to site/models/ (neither is committed).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))

PART = 24 * 2**20  # Cloudflare serves files up to 25 MiB
FEATURES = [  # the model's attribute order, with the page's plain names
    ("Male", "Man"),
    ("Eyeglasses", "Glasses"),
    ("Hat", "Hat"),
    ("Goatee", "Goatee"),
    ("Sideburns", "Sideburns"),
    ("Beard", "Beard"),
    ("Lipstick", "Lipstick"),
    ("Gray_Hair", "Grey hair"),
    ("Bangs", "Bangs"),
    ("Blond_Hair", "Blond hair"),
    ("Smiling", "Smile"),
    ("Mouth_Open", "Open mouth"),
]


def log(*a) -> None:
    print("[web-models]", *a, flush=True)


def drop_accuracy_level(m: onnx.ModelProto) -> onnx.ModelProto:
    """accuracy_level 4 lets the CPU quantise activations to int8 too, which cost the decoder 11 dB: keep float32."""
    for node in m.graph.node:
        if node.op_type == "MatMulNBits":
            for a in [a for a in node.attribute if a.name == "accuracy_level"]:
                node.attribute.remove(a)
    return m


def weight_matmuls(m: onnx.ModelProto) -> list[str]:
    inits = {i.name for i in m.graph.initializer}
    return [n.name for n in m.graph.node if n.op_type == "MatMul" and n.input[1] in inits]


def quantise(src: Path, dst: Path, eight_bit: list[str] | None = None) -> None:
    """All weight MatMuls to 8-bit blocks of 32; or, with `eight_bit`, those to 8-bit and the rest to 4-bit."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    kw = dict(block_size=32, is_symmetric=True, accuracy_level=0)
    model = onnx.load(str(src))
    if eight_bit is None:
        q = MatMulNBitsQuantizer(model, bits=8, **kw)
        q.process()
    else:  # 4-bit everywhere else first, then 8-bit for what is left
        q4 = MatMulNBitsQuantizer(model, bits=4, nodes_to_exclude=eight_bit, **kw)
        q4.process()
        q = MatMulNBitsQuantizer(q4.model.model, bits=8, **kw)
        q.process()
    onnx.save_model(drop_accuracy_level(q.model.model), str(dst), save_as_external_data=False)
    del model, q
    gc.collect()


def pad_conv_channels(m: onnx.ModelProto, multiple: int = 4) -> int:
    """Give every convolution a multiple of 4 input channels (zero channels in, zero weights for them: the same sums).
    ONNX Runtime 1.30's WebGPU convolution miscomputes other counts: TrustMark's encoder has 6 and 70, and its output
    was off by up to 0.14 (the mark then unreadable) until they were padded to 8 and 72."""
    inits = {i.name: i for i in m.graph.initializer}
    nodes, n = [], 0
    for node in m.graph.node:
        group = next((a.i for a in node.attribute if a.name == "group"), 1)
        if node.op_type == "Conv" and group == 1 and node.input[1] in inits:
            w = numpy_helper.to_array(inits[node.input[1]])
            extra = -w.shape[1] % multiple
            if extra:
                pad = np.zeros((w.shape[0], extra, *w.shape[2:]), w.dtype)
                inits[node.input[1]].CopyFrom(
                    numpy_helper.from_array(np.concatenate([w, pad], 1), node.input[1])
                )
                pads = numpy_helper.from_array(
                    np.array([0, 0, 0, 0, 0, extra, 0, 0], np.int64), f"{node.name}_chpads"
                )
                m.graph.initializer.append(pads)
                nodes.append(
                    helper.make_node(
                        "Pad", [node.input[0], pads.name], [f"{node.input[0]}_ch{n}"], mode="constant"
                    )
                )
                node.input[0] = f"{node.input[0]}_ch{n}"
                n += 1
        nodes.append(node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    return n


def half_storage(src: Path, dst: Path) -> None:
    """Store large float32 weights as float16 with a Cast back: half the download, float32 maths once loaded."""
    m = onnx.load(str(src))
    pad_conv_channels(m)
    keep, casts = [], []
    for init in m.graph.initializer:
        a = numpy_helper.to_array(init)
        if a.dtype == np.float32 and a.size >= 1024:
            keep.append(numpy_helper.from_array(a.astype(np.float16), init.name + "_f16"))
            casts.append(
                helper.make_node("Cast", [init.name + "_f16"], [init.name], to=onnx.TensorProto.FLOAT)
            )
        else:
            keep.append(init)
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.save_model(m, str(dst), save_as_external_data=False)


class Raw(nn.Module):
    """The teacher as the sampler calls it: (x_t, t, attrs) -> (x̂ full, x̂ base). Guidance and the ODE run in JS."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x, t, attrs):
        return self.m(x, t, attrs)


def dynamo_export(module: nn.Module, args: tuple, names: list[str], outs: list[str], path: Path) -> None:
    batch = torch.export.Dim("batch", min=1, max=64)
    prog = torch.onnx.export(
        module.eval(),
        args,
        dynamo=True,
        input_names=names,
        output_names=outs,
        dynamic_shapes=tuple({0: batch} for _ in names),
        optimize=True,
        verbose=False,
    )
    n_bytes = sum(t.numel() * t.element_size() for t in [*module.parameters(), *module.buffers()])
    prog.save(str(path), external_data=n_bytes > 1.8e9)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return round(float(10 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12))), 1)


def session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="runs/student_gan_fixed/student.safetensors")
    ap.add_argument("--teacher", default="runs/teacher_b/ckpt/ema.safetensors")
    ap.add_argument("--probe", default="runs/attributes/attribute_probe.pt")
    ap.add_argument("--work", default="onnx/web")
    ap.add_argument("--out", default="site/models")
    ap.add_argument(
        "--skip-check", action="store_true", help="skip the quality measurements (they take ~3 min)"
    )
    a = ap.parse_args()
    work, out = ROOT / a.work, ROOT / a.out
    work.mkdir(parents=True, exist_ok=True)
    cpu = torch.device("cpu")

    from export_onnx import LatentDecoder, OneStepGenerator, export

    from humangen.eval.generate import load_generator
    from humangen.rae import RAE

    # ---- full-precision graphs (reused when already exported from the same checkpoints)
    gen32, dec32, tea32 = (
        ROOT / "onnx/generator_1step.onnx",
        ROOT / "onnx/decoder.onnx",
        work / "teacher_fp32.onnx",
    )
    report = ROOT / "onnx/export_report.json"
    exported = json.loads(report.read_text()).get("student") if report.exists() else None
    if not (gen32.exists() and dec32.exists() and exported and Path(exported) == Path(a.student)):
        log("exporting the generator and decoder")
        model, _, _ = load_generator(ROOT / a.student, cpu)
        k = model.cfg.num_attributes
        z = torch.randn(2, 1024, 16, 16)
        at = torch.zeros(2, k, dtype=torch.long)
        with torch.no_grad():
            export(OneStepGenerator(model.float()).eval(), (z, at), ["noise", "attrs"], gen32)
            export(LatentDecoder(RAE(load_encoder=False)).eval(), (z,), ["latent"], dec32)
        report.write_text(json.dumps({"student": a.student}))
    if not tea32.exists():
        log("exporting the teacher")
        teacher, _, _ = load_generator(ROOT / a.teacher, cpu)
        args = (torch.randn(2, 1024, 16, 16), torch.tensor([0.7, 0.3]), torch.zeros(2, 12, dtype=torch.long))
        with torch.no_grad():
            dynamo_export(Raw(teacher.float()), args, ["x", "t", "attrs"], ["x_full", "x_base"], tea32)
        del teacher
        gc.collect()

    # ---- compressed graphs
    files = {
        "generator": work / "generator_q8.onnx",
        "decoder": work / "decoder_q48.onnx",
        "teacher": work / "teacher_q8.onnx",
        "tm_encoder": work / "tm_encoder_f16.onnx",
        "tm_decoder": work / "tm_decoder_f16.onnx",
    }
    if not files["generator"].exists():
        log("generator -> 8-bit")
        quantise(gen32, files["generator"])
    if not files["decoder"].exists():
        log("decoder -> 8-bit ends, 4-bit middle")
        mm = weight_matmuls(onnx.load(str(dec32), load_external_data=False))
        blocks = [mm[1 + 6 * b : 7 + 6 * b] for b in range((len(mm) - 2) // 6)]
        eight = [mm[0], mm[-1], *(n for blk in blocks[:4] + blocks[-4:] for n in blk)]
        quantise(dec32, files["decoder"], eight)
    if not files["teacher"].exists():
        log("teacher -> 8-bit")
        quantise(tea32, files["teacher"])
    if not (files["tm_encoder"].exists() and files["tm_decoder"].exists()):
        from trustmark import TrustMark

        log("TrustMark -> ONNX, float16 storage")
        tm = TrustMark(verbose=False, model_type="Q", device="cpu", loadRemover=False)

        class Enc(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, cover, secret):
                return self.m(cover, secret)[0]

        class Dec(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, image):
                return self.m.decoder(image)

        for key, mod, args, names, outs in (
            (
                "tm_encoder",
                Enc(tm.encoder),
                (torch.zeros(1, 3, 256, 256), torch.zeros(1, 100)),
                ["cover", "secret"],
                ["stego"],
            ),
            ("tm_decoder", Dec(tm.decoder), (torch.zeros(1, 3, 245, 245),), ["image"], ["bits"]),
        ):
            tmp = work / f"{key}_fp32.onnx"
            if not tmp.exists():
                with torch.no_grad():
                    torch.onnx.export(
                        mod.eval(),
                        args,
                        str(tmp),
                        input_names=names,
                        output_names=outs,
                        opset_version=17,
                        dynamo=False,
                    )
            half_storage(tmp, files[key])

    # ---- small data the page needs: attribute tokens, the checker, the watermark's codeword, the sampler settings
    from watermark import MARK, _tm

    from humangen.data.attributes import INVERT, Probe
    from humangen.diffusion.transport import dimension_shift

    probe = Probe.load(ROOT / a.probe)
    shown = probe.display_names
    order = [shown.index(name) for name, _ in FEATURES]  # the probe's rows, in the model's attribute order
    probe_json = {
        "mu": [round(float(v), 6) for v in probe.mu],
        "sigma": [round(float(v), 6) for v in probe.sigma],
        "weight": [[round(float(v), 7) for v in probe.weight[i]] for i in order],
        "bias": [round(float(probe.bias[i]), 6) for i in order],
        "invert": [probe.names[i] in INVERT for i in order],
    }
    codeword = [int(b) for b in _tm("cpu").ecc.encode_text([MARK])[0]]

    # ---- how close the compressed graphs come to full precision (same inputs, CPU, float32 activations)
    quality = {}
    if not a.skip_check:
        log("measuring quality against full precision")
        rng = np.random.default_rng(0)
        noise = rng.standard_normal((4, 1024, 16, 16)).astype(np.float32)
        attrs = np.zeros((4, 12), dtype=np.int64)
        attrs[1, 1], attrs[2, [0, 10]], attrs[3, [2, 7]] = 2, 2, 1
        g32, g8 = session(gen32), session(files["generator"])
        z32 = g32.run(None, {"noise": noise, "attrs": attrs})[0]
        z8 = g8.run(None, {"noise": noise, "attrs": attrs})[0]
        del g32, g8
        d32, d48 = session(dec32), session(files["decoder"])
        img = lambda d, z: np.concatenate([d.run(None, {"latent": z[i : i + 1]})[0] for i in range(len(z))])  # noqa: E731
        ref, dec_only, both = img(d32, z32), img(d48, z32), img(d48, z8)
        quality["decoder_db"] = [psnr(dec_only[i], ref[i]) for i in range(4)]
        quality["instant_db"] = [psnr(both[i], ref[i]) for i in range(4)]
        quality["note"] = "PSNR of the final 256x256 faces against the full-precision models, same inputs"
        del d32, d48
        gc.collect()

    # ---- parts + manifest
    digest = hashlib.sha256()
    entries = {}
    for key, path in files.items():
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        digest.update(sha.encode())
        entries[key] = {"bytes": len(data), "sha256": sha, "parts": -(-len(data) // PART)}
    version = digest.hexdigest()[:10]
    if out.exists():
        shutil.rmtree(out)
    (out / version).mkdir(parents=True)
    (out / ".gitignore").write_text("*\n")  # hundreds of MB: rebuilt by this script, never committed
    for key, path in files.items():
        data = path.read_bytes()
        for i in range(entries[key]["parts"]):
            (out / version / f"{key}.{i:02d}").write_bytes(data[i * PART : (i + 1) * PART])
    manifest = {
        "version": version,
        "made": time.strftime("%Y-%m-%d"),
        "files": entries,
        "packs": {
            "core": ["generator", "decoder", "tm_encoder"],
            "best": ["teacher"],
            "check": ["tm_decoder"],
        },
        "features": [plain for _, plain in FEATURES],
        "model_features": [name for name, _ in FEATURES],
        "tokens": {"any": 0, "no": 1, "yes": 2},
        "shift": dimension_shift(1024 * 16 * 16),
        "t_eps": 0.05,
        "steps": 25,
        "ig": {"scale": 1.55, "t_min": 0.1, "t_max": 1.0},
        "cfg": {"make": 2.0, "edit": 3.0, "t_min": 0.0, "t_max": 0.7},
        "probe": probe_json,
        "watermark": {"text": MARK, "bits": codeword, "max_flips": 5, "enc_res": 256, "dec_res": 245},
        "quality": quality,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    total = {p: sum(entries[k]["bytes"] for k in ks) / 1e6 for p, ks in manifest["packs"].items()}
    log(f"version {version}: " + ", ".join(f"{p} {mb:.0f} MB" for p, mb in total.items()))
    log(json.dumps(quality))


if __name__ == "__main__":
    main()
