# Runs ON the Colab VM (colab exec -f): rebuild the teacher from its 40 MB parts and check it against the sha256 that
# scripts/colab/stage_inputs.py computed locally. Parts are deleted only after a match.
import glob
import hashlib
import os

IN = "/content/inputs"
with open(f"{IN}/teacher_ema.safetensors.sha256") as f:
    expect = f.read().strip()
parts = sorted(glob.glob(f"{IN}/teacher_ema.safetensors.part*"))
h = hashlib.sha256()
with open(f"{IN}/teacher_ema.safetensors", "wb") as f:
    for p in parts:
        with open(p, "rb") as part:
            b = part.read()
        h.update(b)
        f.write(b)
ok = h.hexdigest() == expect
print(f"{len(parts)} parts, {os.path.getsize(f'{IN}/teacher_ema.safetensors')} bytes, sha256 match: {ok}")
if not ok:
    os.remove(f"{IN}/teacher_ema.safetensors")
    raise SystemExit("teacher checksum mismatch -- re-upload the parts")
for p in parts:
    os.remove(p)
print(sorted(os.listdir(IN)))
