import torch

from humangen.models.dit_dh import HumanDiTDH, make_config
from humangen.optim.cmuon import CMuon, orthogonalized_update, polar_express
from humangen.optim.param_groups import build_optimizer, split_params


def test_polar_express_orthogonalizes():
    torch.manual_seed(0)
    for shape in [(128, 32), (32, 96)]:
        s = torch.linalg.svdvals(polar_express(torch.randn(*shape), steps=5))
        # rectangular Gaussians are well conditioned: all singular values pushed to ~1 in 5 steps
        assert s.min() > 0.8 and s.max() < 1.2, (shape, s.min(), s.max())
    # square Gaussians have near-zero smallest singular values; 5 steps give an approximate
    # orthogonalisation (bulk of the spectrum near 1, nothing blown up), which is all Muon needs
    s = torch.linalg.svdvals(polar_express(torch.randn(64, 64), steps=5))
    assert s.median() > 0.8 and s.max() < 1.2


def test_chunked_update_orthogonalizes_each_chunk():
    torch.manual_seed(0)
    u = torch.randn(6 * 16, 16)  # like an adaLN projection: 6 stacked (16x16) blocks
    out = orthogonalized_update(u, chunks=6, steps=8)
    for block in out.chunk(6, dim=0):
        s = torch.linalg.svdvals(block / (0.2 * 16**0.5))
        assert torch.allclose(s, torch.ones_like(s), atol=0.1)


def test_cmuon_minimizes_quadratic():
    torch.manual_seed(0)
    target = torch.randn(32, 16)
    w = torch.nn.Parameter(torch.zeros(32, 16))
    opt = CMuon([w], lr=0.1)
    first = None
    for step in range(200):
        opt.param_groups[0]["lr"] = 0.1 * (1 - step / 200)  # update RMS is 0.2*lr; decay to settle
        loss = (w - target).pow(2).mean()
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05 * first


def test_param_split_covers_every_parameter_once():
    model = HumanDiTDH(make_config("T"))
    muon, adamw, _ = split_params(model)
    ids = [id(p) for ps in muon.values() for p in ps] + [id(p) for p in adamw]
    assert len(ids) == len(set(ids)) == sum(1 for _ in model.parameters())
    assert 6 in muon and 2 in muon  # head-block adaLN (6 chunks) and final-layer adaLN (2 chunks)


def test_composite_optimizer_steps_and_roundtrips_state():
    torch.manual_seed(0)
    model = HumanDiTDH(make_config("T", num_attributes=3))
    opt = build_optimizer(model, "cmuon", lr=1e-3)
    x = torch.randn(2, 1024, 16, 16)
    before = [p.detach().clone() for p in model.parameters()]
    xf, xb = model(x, torch.rand(2), torch.zeros(2, 3, dtype=torch.long))
    (xf.square().mean() + xb.square().mean() + (xf - x).square().mean()).backward()
    opt.step()
    changed = sum(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert changed > 0
    sd = opt.state_dict()
    opt2 = build_optimizer(model, "cmuon", lr=1e-3)
    opt2.load_state_dict(sd)
