"""Minimal test: find where nGPT diverges under torch.compile vs eager."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'repo'))
import torch
import torch.nn.functional as F

# Test 1: Does normalize_hp work correctly under compile?
def normalize_hp(x):
    dtype = x.dtype
    return (x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(dtype)

x = torch.randn(2, 8, 512, device='cuda', dtype=torch.bfloat16)

eager_out = normalize_hp(x)
compiled_fn = torch.compile(normalize_hp, fullgraph=True)
compile_out = compiled_fn(x)
diff1 = (eager_out - compile_out).abs().max().item()
print(f"Test 1 - normalize_hp: max diff = {diff1:.2e} {'PASS' if diff1 < 1e-6 else 'FAIL'}")

# Test 2: Does the interpolation diverge?
def interpolate_norm(x, update, alpha=0.05):
    return normalize_hp(x + alpha * (update - x))

update = torch.randn_like(x)
eager_out2 = interpolate_norm(x, update)
compiled_fn2 = torch.compile(interpolate_norm, fullgraph=True)
compile_out2 = compiled_fn2(x, update)
diff2 = (eager_out2 - compile_out2).abs().max().item()
print(f"Test 2 - interpolate+norm: max diff = {diff2:.2e} {'PASS' if diff2 < 1e-6 else 'FAIL'}")

# Test 3: Chain of normalize-interpolate (simulating nGPT layers)
def chain_norm(x, n_layers=11):
    for i in range(n_layers):
        x_norm = normalize_hp(x)
        # Simulate attention: random linear + normalize
        w = torch.randn(512, 512, device=x.device, dtype=x.dtype) * 0.01
        attn_out = F.linear(x_norm, w)
        alpha = torch.full((512,), 0.05, device=x.device, dtype=x.dtype)
        x = normalize_hp(x_norm + alpha * (attn_out - x_norm))
        # Simulate MLP
        w2 = torch.randn(512, 1536, device=x.device, dtype=x.dtype) * 0.01
        w3 = torch.randn(512, 1536, device=x.device, dtype=x.dtype) * 0.01
        mlp_out = F.linear(F.leaky_relu(F.linear(x, w2.t()), 0.5).square(), w3)
        x = normalize_hp(x + alpha * (mlp_out - x))
    return x

torch.manual_seed(42)
x3 = torch.randn(1, 32, 512, device='cuda', dtype=torch.bfloat16)

torch.manual_seed(42)
eager_out3 = chain_norm(x3.clone())

torch.manual_seed(42)
compiled_fn3 = torch.compile(chain_norm, fullgraph=True)
compile_out3 = compiled_fn3(x3.clone())

diff3 = (eager_out3 - compile_out3).abs().max().item()
cos_sim = F.cosine_similarity(eager_out3.reshape(-1, 512), compile_out3.reshape(-1, 512), dim=-1).mean().item()
print(f"Test 3 - 11-layer chain: max diff = {diff3:.2e}, cos_sim = {cos_sim:.6f} {'PASS' if diff3 < 0.01 else 'FAIL'}")

# Test 4: Same but with emulate_precision_casts
torch._inductor.config.emulate_precision_casts = True
torch.manual_seed(42)
compiled_fn4 = torch.compile(chain_norm, fullgraph=True)
compile_out4 = compiled_fn4(x3.clone())
diff4 = (eager_out3 - compile_out4).abs().max().item()
cos_sim4 = F.cosine_similarity(eager_out3.reshape(-1, 512), compile_out4.reshape(-1, 512), dim=-1).mean().item()
print(f"Test 4 - chain + precision_casts: max diff = {diff4:.2e}, cos_sim = {cos_sim4:.6f} {'PASS' if diff4 < 0.01 else 'FAIL'}")

# Test 5: What if we do everything in float32?
def chain_norm_fp32(x, n_layers=11):
    x = x.float()
    for i in range(n_layers):
        x_norm = F.normalize(x, dim=-1)
        w = torch.randn(512, 512, device=x.device, dtype=torch.float32) * 0.01
        attn_out = F.linear(x_norm, w)
        alpha = 0.05
        x = F.normalize(x_norm + alpha * (attn_out - x_norm), dim=-1)
        w2 = torch.randn(1536, 512, device=x.device, dtype=torch.float32) * 0.01
        w3 = torch.randn(512, 1536, device=x.device, dtype=torch.float32) * 0.01
        mlp_out = F.linear(F.leaky_relu(F.linear(x, w2), 0.5).square(), w3)
        x = F.normalize(x + alpha * (mlp_out - x), dim=-1)
    return x

torch.manual_seed(42)
eager_fp32 = chain_norm_fp32(x3.clone())
torch.manual_seed(42)
compiled_fp32 = torch.compile(chain_norm_fp32, fullgraph=True)(x3.clone())
diff5 = (eager_fp32 - compiled_fp32).abs().max().item()
print(f"Test 5 - fp32 chain: max diff = {diff5:.2e} {'PASS' if diff5 < 1e-5 else 'FAIL'}")

print("\nIf Test 3 FAILS but Test 5 PASSES: the issue is bf16 precision under compile")
print("If both FAIL: the issue is op reordering/fusion, not precision")

# Test 6: Opaque L2NormalizeHP under compile with bf16 autocast
@torch._dynamo.allow_in_graph
class L2NormalizeHP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        dtype = x.dtype
        x32 = x.float()
        norm = x32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y = (x32 / norm).to(dtype)
        ctx.save_for_backward(y, norm.reciprocal().to(torch.float32))
        return y
    @staticmethod
    def backward(ctx, dy):
        y, inv_norm = ctx.saved_tensors
        dy32, y32 = dy.float(), y.float()
        dot = (dy32 * y32).sum(-1, keepdim=True)
        return ((dy32 - y32 * dot) * inv_norm).to(dy.dtype)

def chain_norm_opaque(x, n_layers=11):
    for i in range(n_layers):
        x_norm = L2NormalizeHP.apply(x)
        w = torch.randn(512, 512, device=x.device, dtype=x.dtype) * 0.01
        attn_out = F.linear(x_norm, w)
        alpha = torch.full((512,), 0.05, device=x.device, dtype=x.dtype)
        x = L2NormalizeHP.apply(x_norm + alpha * (attn_out - x_norm))
        w2 = torch.randn(512, 1536, device=x.device, dtype=x.dtype) * 0.01
        w3 = torch.randn(512, 1536, device=x.device, dtype=x.dtype) * 0.01
        mlp_out = F.linear(F.leaky_relu(F.linear(x, w2.t()), 0.5).square(), w3)
        x = L2NormalizeHP.apply(x + alpha * (mlp_out - x))
    return x

print("\n--- Test 6: Opaque normalize (allow_in_graph) ---")
torch._dynamo.reset()
torch.manual_seed(42)
x6 = torch.randn(1, 32, 512, device='cuda', dtype=torch.bfloat16)

torch.manual_seed(42)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    eager_out6 = chain_norm_opaque(x6.clone())

torch.manual_seed(42)
compiled_opaque = torch.compile(chain_norm_opaque)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    compile_out6 = compiled_opaque(x6.clone())

diff6 = (eager_out6 - compile_out6).abs().max().item()
cos_sim6 = F.cosine_similarity(eager_out6.reshape(-1, 512), compile_out6.reshape(-1, 512), dim=-1).mean().item()
print(f"Test 6 - opaque normalize: max diff = {diff6:.2e}, cos_sim = {cos_sim6:.6f} {'PASS' if cos_sim6 > 0.999 else 'FAIL'}")

# Test 7: Gradient correctness
print("\n--- Test 7: Gradient check ---")
x7 = torch.randn(2, 4, 64, dtype=torch.float64, requires_grad=True, device='cuda')
try:
    torch.autograd.gradcheck(L2NormalizeHP.apply, (x7,), eps=1e-4, atol=1e-3, rtol=1e-3)
    print("Test 7 - gradcheck: PASS")
except Exception as e:
    print(f"Test 7 - gradcheck: FAIL ({e})")
