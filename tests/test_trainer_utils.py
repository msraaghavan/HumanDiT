import torch

from humangen.data.latents import drop_attributes
from humangen.models.layers import AttributeTokens
from humangen.train.trainer import ema_beta, lr_factor


def test_lr_schedule_matches_raev2_shape():
    total = 800
    f = [lr_factor(s, total, 0.3125, 0.625, 0.1) for s in range(total)]
    assert f[0] == 1.0 and f[249] == 1.0  # constant phase (25/80 of training)
    assert abs(f[375] - 0.55) < 0.01  # halfway through the linear decay
    assert f[500] == 0.1 and f[-1] == 0.1  # final plateau from 50/80


def test_ema_halflife():
    beta = ema_beta(batch_size=256, halflife_samples=256 * 100)
    assert abs(beta**100 - 0.5) < 1e-9


def test_attribute_dropout_rates():
    torch.manual_seed(0)
    a = torch.full((20000, 6), AttributeTokens.YES)
    d = drop_attributes(a, p_all=0.1, p_each=0.3)
    all_null = (d == AttributeTokens.NULL).all(1).float().mean().item()
    frac_null = (d == AttributeTokens.NULL).float().mean().item()
    assert abs(all_null - (0.1 + 0.9 * 0.3**6)) < 0.01
    assert abs(frac_null - (0.1 + 0.9 * 0.3)) < 0.01
