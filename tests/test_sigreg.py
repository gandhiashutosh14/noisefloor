import torch

from noisefloor.models.sigreg import effective_rank, sigreg


def _g(seed=0):
    return torch.Generator().manual_seed(seed)


def test_sigreg_is_small_on_isotropic_gaussian_and_large_on_collapse():
    g = _g()
    iso = torch.randn(2048, 16, generator=g)
    collapsed = torch.zeros(2048, 16) + 0.3
    low_rank = torch.randn(2048, 1, generator=g) @ torch.randn(1, 16, generator=g)
    scaled = 4.0 * torch.randn(2048, 16, generator=g)
    s_iso = float(sigreg(iso, generator=_g(1)))
    # N * EP has expectation int (1 - e^{-t^2}) e^{-t^2/2} dt = sqrt(2 pi) - sqrt(2 pi / 3) ~ 1.06 under the null
    assert 0.5 < s_iso < 2.0
    for bad in (collapsed, low_rank, scaled):
        assert float(sigreg(bad, generator=_g(1))) > 20 * s_iso


def test_sigreg_gradient_moves_a_collapsed_code_toward_isotropy():
    torch.manual_seed(0)
    z = (0.05 * torch.randn(1024, 8)).requires_grad_()
    opt = torch.optim.SGD([z], lr=5.0)
    start = float(sigreg(z, generator=_g(2)))
    for step in range(150):
        opt.zero_grad()
        loss = sigreg(z, generator=_g(step))
        loss.backward()
        opt.step()
    end = float(sigreg(z, generator=_g(2)))
    assert end < 0.2 * start
    assert effective_rank(z.detach()) > 6.0


def test_sigreg_is_batch_size_consistent_and_rejects_bad_shapes():
    g = _g(3)
    small = float(sigreg(torch.randn(512, 8, generator=g), generator=_g(4)))
    large = float(sigreg(torch.randn(4096, 8, generator=g), generator=_g(4)))
    assert abs(small - large) < 0.5     # N * EP is O(1) under the null
    try:
        sigreg(torch.randn(8))
    except ValueError:
        pass
    else:
        raise AssertionError("1-D input accepted")


def test_sigreg_float64_gradcheck():
    z = torch.randn(64, 4, dtype=torch.float64, generator=_g(6)).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: sigreg(x, num_slices=8, generator=_g(7)), (z,))


def test_effective_rank_bounds():
    g = _g(5)
    assert effective_rank(torch.randn(4096, 8, generator=g)) > 7.5
    assert effective_rank(torch.randn(4096, 1, generator=g) @ torch.randn(1, 8, generator=g)) < 1.5
