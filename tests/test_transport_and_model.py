import torch

from humangen.diffusion.guidance import GuidanceConfig, guided_model_fn
from humangen.diffusion.transport import Transport, dimension_shift, shift_t
from humangen.models.dit_dh import HumanDiTDH, make_config


def test_dimension_shift_matches_raev2():
    assert dimension_shift(1024 * 16 * 16) == 8.0


def test_shift_is_monotone_and_fixes_endpoints():
    t = torch.linspace(0, 1, 101)
    s = shift_t(t, 8.0)
    assert s[0] == 0 and torch.isclose(s[-1], torch.tensor(1.0))
    assert (s[1:] >= s[:-1]).all() and (s >= t).all()  # shift > 1 spends more time at high noise


def test_oracle_x_prediction_has_zero_loss():
    tr = Transport()
    x1 = torch.randn(4, 8, 4, 4)
    out = tr.loss(lambda x_t, t: (x1, x1), x1)
    assert out["loss"].item() < 1e-8


def test_sampler_recovers_target_with_oracle():
    tr = Transport()
    target = torch.randn(2, 8, 4, 4)
    for method in ("euler", "heun"):
        x = tr.sample(lambda x, t: target, torch.randn(2, 8, 4, 4), num_steps=10, method=method)
        assert torch.allclose(x, target, atol=1e-5), method


def test_model_shapes_and_zero_init():
    cfg = make_config("T", num_attributes=5)
    m = HumanDiTDH(cfg)
    x = torch.randn(3, 1024, 16, 16)
    attrs = torch.randint(0, 3, (3, 5))
    xf, xb = m(x, torch.rand(3), attrs)
    assert xf.shape == xb.shape == x.shape
    # adaLN-Zero + zero output layers -> exactly zero prediction at initialisation
    assert xf.abs().max() == 0 and xb.abs().max() == 0


def test_guidance_formula():
    class Fake(torch.nn.Module):
        def forward(self, x, t, attrs):
            cond = (attrs != 0).any(1).float().view(-1, 1, 1, 1)
            return x + 1 + cond, x  # full, base

    attrs = torch.tensor([[2, 0]])
    x, t = torch.zeros(1, 1, 2, 2), torch.tensor([0.5])
    g = GuidanceConfig(ig_scale=2.0, ig_t_min=0.0, ig_t_max=1.0, cfg_scale=3.0, cfg_t_min=0.0, cfg_t_max=1.0)
    out = guided_model_fn(Fake(), attrs, g)(x, t)
    ig_c, ig_u = 2.0 * 2, 2.0 * 1  # base + w*(full-base) with base = 0
    assert torch.allclose(out, torch.full_like(x, ig_u + 3.0 * (ig_c - ig_u)))
