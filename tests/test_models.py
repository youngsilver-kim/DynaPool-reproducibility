import pytest
import torch

from dynapool.models import GeMPooling, SmallImageResNet18, build_model, method_spec


def test_small_image_stem_keeps_eight_by_eight_final_map():
    model = SmallImageResNet18().eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 512, 8, 8)


@pytest.mark.parametrize(
    "method",
    [
        "avg",
        "max",
        "gem",
        "att",
        "dyna",
        "dyna_entropy",
        "equal",
        "static_mean",
        "dyna_att_only",
        "dyna_drop_att",
        "att_avg",
    ],
)
def test_all_heads_return_valid_logits_and_coefficients(method):
    weights = [0.1, 0.2, 0.3, 0.4] if method == "static_mean" else None
    model = build_model(method, num_classes=7, static_weights=weights).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))
    assert output["logits"].shape == (2, 7)
    if method not in {"avg", "max", "gem", "att"}:
        assert torch.allclose(output["alpha"].sum(dim=1), torch.ones(2), atol=1e-6)
        assert torch.all(output["alpha"] >= 0)


def test_gem_exponent_is_strictly_positive_and_trainable():
    pool = GeMPooling(p_init=3.0, p_min=1e-3)
    assert pool.p.item() == pytest.approx(3.0, rel=1e-5)
    result = pool(torch.rand(2, 4, 8, 8, requires_grad=True))
    result.sum().backward()
    assert pool.raw_p.grad is not None
    assert torch.isfinite(pool.raw_p.grad)


def test_branch_removal_specs_really_remove_requested_branch():
    active, gate_mode = method_spec("dyna_drop_att")
    assert active == ("avg", "max", "gem")
    assert gate_mode == "dynamic"
