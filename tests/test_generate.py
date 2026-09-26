import contextlib

import torch

from humangen.eval.generate import iterate_latents


def test_iterate_latents_does_not_leak_context_into_caller():
    """Grad mode / autocast are thread-global: yielding inside them would leak into the caller's code."""
    entered = []

    @contextlib.contextmanager
    def fake_autocast():
        entered.append(True)
        torch.set_grad_enabled(False)  # stand-in for a global mode switch
        try:
            yield
        finally:
            torch.set_grad_enabled(True)

    def sample_fn(noise, attrs):
        assert not torch.is_grad_enabled()
        return noise[:, :1]

    with torch.enable_grad():
        for z, a in iterate_latents(
            sample_fn, n=5, batch=2, num_attributes=3, device=torch.device("cpu"), autocast=fake_autocast
        ):
            assert torch.is_grad_enabled(), "caller code must run outside the sampling context"
            assert z.shape[0] == a.shape[0] <= 2
    assert len(entered) == 3


def test_iterate_latents_is_deterministic():
    def fn(noise, attrs):
        return noise

    a = torch.cat([z for z, _ in iterate_latents(fn, 6, 4, 2, torch.device("cpu"), seed=3)])
    b = torch.cat([z for z, _ in iterate_latents(fn, 6, 4, 2, torch.device("cpu"), seed=3)])
    assert torch.equal(a, b)
