# Model card: HumanDiT

## Summary
HumanDiT generates **synthetic** 256×256 human faces, optionally conditioned on up to 12 face attributes. It is a
flow-matching diffusion transformer (DiT^DH-B, 183M parameters) working in the RAEv2 latent space, which consists
of frozen DINOv3-L features plus a ViT-XL pixel decoder. It is paired with one-step students (W-Flow and adversarial).

## Intended use
Research, education and portfolio demonstration of generative-model engineering. **Not for commercial use.**

## Out-of-scope and prohibited use
- Impersonating, identifying or making claims about real people. Generated faces are synthetic, but they may resemble
  real individuals by chance.
- Creating deceptive media (fake profiles, fraud, harassment, political disinformation).
- Any decision about people (hiring, lending, policing, biometric verification).

## Training data
- **FFHQ** (70,000 Flickr faces, CC BY-NC-SA 4.0), 256×256, with horizontal flips.
- **CelebA** attribute annotations (non-commercial research agreement), used only to train a linear probe that
  pseudo-labels FFHQ.

## Attribute labels
The probe abstains when uncertain: thresholds are calibrated on CelebA validation data for ≥90% precision
("present") and ≥97% negative predictive value ("absent"). Subjective or appearance-judgement attributes (for example
"Attractive") and skin tone are excluded. CelebA's binary "Male" label is a simplification of gender presentation
and does not represent gender identity.

## Known biases and limitations
- FFHQ over-represents some age groups, ethnicities and photographic styles. Under-represented groups are likely to
  be generated with lower fidelity and less often. We report unconditional attribute base rates in the README.
- Attribute pseudo-labels inherit CelebA's annotation biases and the CelebA→FFHQ domain shift.
- Quality is measured with FID, FD_DINOv2 and precision/recall, which capture only part of perceptual quality.

## Safety measures
- An invisible DWT-DCT watermark and "AI-generated" PNG metadata are added to every demo output, and the demo includes a watermark checker.
- A memorisation audit compares 10k samples with all 70k training faces in DINOv3 space; results are in the README.

## Licences of components
Code: MIT. Files adapted from RAEv2: CC BY-NC 4.0. W-Flow loss ported from the MIT-licensed official code.
DINOv3 weights: DINOv3 License. Trained weights: non-commercial (CC BY-NC-SA 4.0, inherited from FFHQ).
