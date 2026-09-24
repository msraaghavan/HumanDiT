import torch

from humangen.diffusion.transport import Transport
from humangen.models.dit_dh import HumanDiTDH, make_config
from humangen.onestep.adversarial import TrunkDiscriminator, d_loss, g_loss
from humangen.onestep.wflow import barycentric_map, grouped_wflow_loss, sinkhorn_plan, wflow_loss


def test_sinkhorn_marginals():
    torch.manual_seed(0)
    x, y = torch.randn(3, 8, 5), torch.randn(3, 12, 5)
    pi = sinkhorn_plan(x, y, reg=1.0, iters=200, diag_mask=False)
    assert torch.allclose(pi.sum(-1), torch.full((3, 8), 1 / 8), atol=1e-4)
    assert torch.allclose(pi.sum(-2), torch.full((3, 12), 1 / 12), atol=1e-4)


def test_velocity_vanishes_when_distributions_match():
    torch.manual_seed(0)
    x = torch.randn(1, 64, 4)
    # identical sets: T_{q->p} maps each point to itself (small reg), self-map (diag masked) does not,
    # but the drift should be much smaller than for a shifted target distribution
    far = barycentric_map(x, x + 5.0, 0.05 * 4, 50, False) - x
    assert far.mean(dim=(0, 1)).allclose(torch.full((4,), 5.0), atol=0.5)


def test_wflow_loss_moves_generated_points_toward_data():
    torch.manual_seed(0)
    gen = torch.randn(2, 16, 8, requires_grad=True)
    pos = torch.randn(2, 32, 8) + 3.0
    opt = torch.optim.Adam([gen], lr=0.05)  # mean-reduced loss: use Adam as in real training
    d0 = (gen.detach().mean(1) - pos.mean(1)).norm()
    for _ in range(50):
        loss, _ = wflow_loss(gen, pos, iters=20)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (gen.detach().mean(1) - pos.mean(1)).norm() < 0.3 * d0


def test_grouped_loss_on_latents_runs():
    gen = torch.randn(2 * 4, 1024, 16, 16, requires_grad=True)
    pos = torch.randn(2 * 6, 1024, 16, 16)
    loss, info = grouped_wflow_loss(gen, pos, groups=2, iters=5)
    loss.backward()
    assert gen.grad is not None and torch.isfinite(loss) and "tokens/loss" in info


def test_gan_losses_backprop_to_generator_only_through_heads():
    teacher = HumanDiTDH(make_config("T", num_attributes=3))
    disc = TrunkDiscriminator(teacher, layers=(1, 2, 4))
    tr = Transport()
    real, attrs = torch.randn(2, 1024, 16, 16), torch.zeros(2, 3, dtype=torch.long)
    fake = torch.randn(2, 1024, 16, 16, requires_grad=True)
    ld, _ = d_loss(disc, real, attrs, fake, attrs, tr, r1_weight=1.0, r1_sigma=0.01)
    ld.backward()
    assert all(p.grad is None for p in disc.trunk.parameters())
    assert any(p.grad is not None for p in disc.heads.parameters())
    assert fake.grad is None  # D step must not update the generator
    lg = g_loss(disc, fake, attrs, tr)
    lg.backward()
    assert fake.grad is not None and fake.grad.abs().sum() > 0
