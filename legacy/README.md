# Legacy: the original Human GAN (kept for before/after)

`*_original.py` are the files exactly as they were. `dcgan.py` is the same DCGAN with only its bugs fixed, and is
used as the baseline.

What was wrong with the original:

| Issue | Where | Effect |
|---|---|---|
| `Sigmoid` + `BCELoss` | `dcgan_original.py` | numerically unstable; `BCEWithLogitsLoss` is the standard fix |
| Unreadable images replaced by `torch.zeros` | dataset `__getitem__` | silently trains the discriminator on black images |
| No `if __name__ == "__main__"` guard | all scripts | `num_workers > 0` crashes on Windows |
| "Latent consistency loss" `|G(z) - G(z + 0.1ε)|` | `hclgan_original.py` | penalises output *diversity*, pushing towards mode collapse (the opposite of MSGAN) |
| "Latency-consistent" skip connections added after `Tanh` | `hlc_gan_original.py` | outputs leave [-1, 1]; "latency consistency" has no meaning here; model never trained |
| README promises `train.py`, `generate_faces.py`, "highly realistic" faces | `README_original.md` | none of these exist |
| Duplicate `Human_Gan-main/` folder, CPU-only torch | repo | no GPU was ever used |
