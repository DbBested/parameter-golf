"""Test Triton fused normalize kernel correctness and performance."""
import os
os.environ["TRITON_NORM"] = "1"
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'repo'))
import time
import torch
import torch.nn.functional as F

# Import after setting env var
from train_gpt import L2NormalizeHP, _HAS_TRITON

print(f"Triton available: {_HAS_TRITON}")

# Test 1: Correctness (bf16)
print("\n=== Test 1: Correctness (bf16, 3D) ===")
x = torch.randn(4, 128, 512, device='cuda', dtype=torch.bfloat16)
ref = F.normalize(x.float(), dim=-1).to(torch.bfloat16)
out = L2NormalizeHP.apply(x)
diff = (ref - out).abs().max().item()
cos = F.cosine_similarity(ref.reshape(-1, 512), out.reshape(-1, 512), dim=-1).mean().item()
print(f"  max diff = {diff:.2e}, cos_sim = {cos:.8f} {'PASS' if diff < 1e-3 else 'FAIL'}")

# Test 2: Correctness (bf16, 2D - weight matrix)
print("\n=== Test 2: Correctness (bf16, 2D weight) ===")
w = torch.randn(1536, 512, device='cuda', dtype=torch.bfloat16)
ref_w = F.normalize(w.float(), dim=-1).to(torch.bfloat16)
out_w = L2NormalizeHP.apply(w)
diff_w = (ref_w - out_w).abs().max().item()
print(f"  max diff = {diff_w:.2e} {'PASS' if diff_w < 1e-3 else 'FAIL'}")

# Test 3: Gradient correctness
print("\n=== Test 3: Gradient check ===")
x_grad = torch.randn(2, 4, 64, device='cuda', dtype=torch.float64, requires_grad=True)
try:
    # Triton kernel only for bf16, so gradcheck with fp64 uses PyTorch fallback
    torch.autograd.gradcheck(L2NormalizeHP.apply, (x_grad,), eps=1e-4, atol=1e-3, rtol=1e-3)
    print("  gradcheck (fp64 fallback): PASS")
except Exception as e:
    print(f"  gradcheck: FAIL ({e})")

# Test 4: bf16 backward correctness
print("\n=== Test 4: bf16 backward ===")
x4 = torch.randn(4, 32, 512, device='cuda', dtype=torch.bfloat16, requires_grad=True)
y4 = L2NormalizeHP.apply(x4)
loss = y4.sum()
loss.backward()
grad_triton = x4.grad.clone()

x4_ref = x4.detach().clone().requires_grad_(True)
y4_ref = F.normalize(x4_ref.float(), dim=-1).to(torch.bfloat16)
loss_ref = y4_ref.sum()
loss_ref.backward()
grad_ref = x4_ref.grad

diff_grad = (grad_triton.float() - grad_ref.float()).abs().max().item()
cos_grad = F.cosine_similarity(grad_triton.reshape(-1, 512).float(), grad_ref.reshape(-1, 512).float(), dim=-1).mean().item()
print(f"  grad max diff = {diff_grad:.2e}, cos_sim = {cos_grad:.8f} {'PASS' if cos_grad > 0.999 else 'FAIL'}")

# Test 5: Performance benchmark
print("\n=== Test 5: Performance (4, 384, 512) ===")
x5 = torch.randn(4, 384, 512, device='cuda', dtype=torch.bfloat16)
# Warmup
for _ in range(10):
    L2NormalizeHP.apply(x5)
torch.cuda.synchronize()

# Triton
N = 100
t0 = time.perf_counter()
for _ in range(N):
    L2NormalizeHP.apply(x5)
torch.cuda.synchronize()
t_triton = (time.perf_counter() - t0) / N * 1000

# PyTorch reference (opaque function without Triton)
os.environ["TRITON_NORM"] = "0"
t0 = time.perf_counter()
for _ in range(N):
    dtype = x5.dtype
    x32 = x5.float()
    norm = x32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y = (x32 / norm).to(dtype)
torch.cuda.synchronize()
t_pytorch = (time.perf_counter() - t0) / N * 1000

print(f"  Triton:  {t_triton:.3f} ms")
print(f"  PyTorch: {t_pytorch:.3f} ms")
print(f"  Speedup: {t_pytorch/t_triton:.2f}x")
print(f"  Savings per forward (86 calls): {(t_pytorch - t_triton) * 86:.1f} ms")
