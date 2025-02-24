import pytest
import torch

from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3MLP
from transformers.models.granitemoeshared.configuration_granitemoeshared import GraniteMoeSharedConfig
from transformers.models.granitemoeshared.modeling_granitemoeshared import GraniteMoeSharedMLP

from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.transformers.functional import liger_swiglu
from liger_kernel.transformers.swiglu import LigerPhi3SwiGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.swiglu import LigerGraniteMoeSharedMoESwiGLUMLP

from liger_kernel.utils import infer_device

device = infer_device()

LLAMA_CONFIG = LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    hidden_act="silu",
)
PHI3_CONFIG = Phi3Config(
    hidden_size=4096,
    intermediate_size=11008,
    hidden_act="silu",
)
GRANITEMOESHARED_CONFIG = GraniteMoeSharedConfig(
    hidden_size=4096,
    intermediate_size=11008,
    num_local_experts=4,  # Example value, adjust as needed
    num_experts_per_tok=2,  # Example value, adjust as needed
    hidden_act="silu",
)
SLEEP_SECONDS = 0.1



@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 256, 256, 512),
        # weird shapes
        (6, 42, 123, 431),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-0, 1e-5),
        # TODO: we should find a better way to tune this. 1e4 is too large apparently
        pytest.param(
            torch.bfloat16,
            1e4,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness_llamamlp(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    G = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    U = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)

    llama_mlp = LlamaMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    llama_mlp.gate_proj.weight.data = G.T
    llama_mlp.up_proj.weight.data = U.T
    llama_mlp.down_proj.weight.data = D.T

    liger_mlp = LigerSwiGLUMLP(config=LLAMA_CONFIG).to(device).to(dtype)
    liger_mlp.gate_proj.weight.data = G.T
    liger_mlp.up_proj.weight.data = U.T
    liger_mlp.down_proj.weight.data = D.T

    y1 = llama_mlp(x1)
    y2 = liger_mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    assert torch.allclose(
        llama_mlp.gate_proj.weight.grad,
        liger_mlp.gate_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        llama_mlp.up_proj.weight.grad,
        liger_mlp.up_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        llama_mlp.down_proj.weight.grad,
        liger_mlp.down_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 256, 256, 512),
        # weird shapes
        (6, 42, 123, 431),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-0, 1e-5),
        # TODO: we should find a better way to tune this. 1e4 is too large apparently
        pytest.param(
            torch.bfloat16,
            1e4,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness_phi3mlp(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    GU = torch.randn(hidden_size, intermediate_size * 2, device=device, dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)

    phi3_mlp = Phi3MLP(config=PHI3_CONFIG).to(device).to(dtype)
    phi3_mlp.gate_up_proj.weight.data = GU.T
    phi3_mlp.down_proj.weight.data = D.T

    liger_mlp = LigerPhi3SwiGLUMLP(config=PHI3_CONFIG).to(device).to(dtype)
    liger_mlp.gate_up_proj.weight.data = GU.T
    liger_mlp.down_proj.weight.data = D.T

    y1 = phi3_mlp(x1)
    y2 = liger_mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    assert torch.allclose(
        phi3_mlp.gate_up_proj.weight.grad,
        liger_mlp.gate_up_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        phi3_mlp.down_proj.weight.grad,
        liger_mlp.down_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, size",
    [
        (2, 8, 8),
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-0, 1e-5),
        # TODO: we should find a better way to tune this. 1e4 is too large apparently
        (torch.bfloat16, 1e4, 1e-2),
    ],
)



def test_correctness_sharedgranitemoe(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)
    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    # initialize weights
    fake_input_linear_weight = torch.randn(GRANITEMOESHARED_CONFIG.num_local_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype)
    fake_output_linear_weight = torch.randn(GRANITEMOESHARED_CONFIG.num_local_experts, hidden_size, intermediate_size, device=device, dtype=dtype)
    fake_router_weight = torch.randn(hidden_size, GRANITEMOESHARED_CONFIG.num_local_experts, device=device, dtype=dtype)

    gmoes_mlp = GraniteMoeSharedMLP(config=GRANITEMOESHARED_CONFIG).to(device).to(dtype)
    gmoes_mlp.input_linear.weight.data = fake_input_linear_weight
    gmoes_mlp.output_linear.weight.data = fake_input_linear_weight
    gmoes_mlp.router.weight.data = fake_router_weight

    liger_mlp = LigerGraniteMoeSharedMoESwiGLUMLP(config=GRANITEMOESHARED_CONFIG).to(device).to(dtype)
    liger_mlp.input_linear.weight.data = fake_input_linear_weight
    liger_mlp.output_linear.weight.data = fake_input_linear_weight
    liger_mlp.router.weight.data = fake_router_weight

    y1 = gmoes_mlp(x1)
    y2 = liger_mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    assert torch.allclose(
        gmoes_mlp.input_linear.weight.grad,
        liger_mlp.input_linear.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        gmoes_mlp.output_linear.weight.grad,
        liger_mlp.output_linear.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        gmoes_mlp.router.weight.grad,
        liger_mlp.router.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, size",
    [
        (2, 8, 8),
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol is for small values: they have more difference, so set atol higher
        # rtol is for larger values: they are very close, so set rtol lower
        (torch.float32, 1e-0, 1e-5),
        # TODO: we should find a better way to tune this. 1e4 is too large apparently
        (torch.bfloat16, 1e4, 1e-2),
    ],
)

def test_correctness_functional(bsz, seq_len, size, dtype, atol, rtol):
    _input = torch.randn(bsz, seq_len, size, device=device, dtype=dtype)
    _b = torch.randn(bsz, seq_len, size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    b1 = _b.clone().requires_grad_(True)
    b2 = _b.clone().requires_grad_(True)

    y1 = liger_swiglu(a=x1, b=b1)
    y2 = LigerSiLUMulFunction.apply(x2, b2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    # Test backward pass
    grad_output = torch.randn_like(y1)

    y1.backward(grad_output)
    y2.backward(grad_output)

    # Check if gradients are close for x
    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(b1.grad, b2.grad, atol=atol, rtol=rtol)