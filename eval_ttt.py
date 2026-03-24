"""Standalone TTT eval — loads a saved quantized model and runs TTT evaluation.
Usage: torchrun --nproc_per_node=N eval_ttt.py
"""
import os, sys, time, io, math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import zstandard
import sentencepiece as spm

from train_gpt import (
    Hyperparameters, GPT, CastedLinear,
    load_validation_tokens, build_sentencepiece_luts,
    eval_val_sliding, score_first_ttt, eval_val_sliding_with_temperature,
    dequantize_mixed_int6, restore_low_dim_params_to_fp32,
)

rank = int(os.environ.get('RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '1'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
device = torch.device(f'cuda:{local_rank}')
torch.cuda.set_device(device)

distributed = world_size > 1
if distributed:
    dist.init_process_group(backend='nccl')

def log0(s):
    if rank == 0:
        print(s, flush=True)

args = Hyperparameters()

log0('Loading validation tokens...')
val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
log0(f'val_tokens: {val_tokens.numel()-1} tokens')

sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
    sp, args.vocab_size, device
)

log0('Loading quantized model...')
model_path = os.environ.get('MODEL_PATH', 'final_model.int6.gptqlite.ptz')
if not os.path.exists(model_path):
    model_path = 'final_model.int6.ptz'
log0(f'Using model: {model_path}')
with open(model_path, 'rb') as f:
    quant_blob = f.read()
quant_state = torch.load(
    io.BytesIO(zstandard.ZstdDecompressor().decompress(quant_blob)),
    map_location='cpu',
)

sd_template = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
    ve_layers=args.ve_layers, vrl_enabled=True,
).state_dict()

deq_state = dequantize_mixed_int6(quant_state['w'], quant_state['m'], sd_template)

eval_model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
    ve_layers=args.ve_layers, vrl_enabled=True,
).to(device).bfloat16()

for m in eval_model.modules():
    if isinstance(m, CastedLinear):
        m.float()
restore_low_dim_params_to_fp32(eval_model)
eval_model.load_state_dict(deq_state, strict=True)
log0('Model loaded.')

# 1. Baseline sliding window
log0('=== Baseline sliding window eval (no TTT) ===')
t0 = time.perf_counter()
sw_loss, sw_bpb = eval_val_sliding(
    args, eval_model, rank, world_size, device,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride=64,
)
log0(f'baseline val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} time:{time.perf_counter()-t0:.0f}s')

# 2. TTT score-first
log0('=== Score-first TTT ===')
t1 = time.perf_counter()
ttt_loss, ttt_bpb = score_first_ttt(
    args, eval_model, rank, world_size, device,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
)
log0(f'ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} time:{time.perf_counter()-t1:.0f}s')

# 3. Re-score at T=0.98
log0('=== Post-TTT re-score at T=0.98 ===')
t2 = time.perf_counter()
t_loss, t_bpb = eval_val_sliding_with_temperature(
    args, eval_model, rank, world_size, device,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride=64, temperature=0.98,
)
log0(f'ttt+T0.98 val_loss:{t_loss:.4f} val_bpb:{t_bpb:.4f} time:{time.perf_counter()-t2:.0f}s')

log0(f'=== SUMMARY ===')
log0(f'Baseline (sliding s=64): {sw_bpb:.4f}')
log0(f'After TTT (score-first): {ttt_bpb:.4f} (delta: {ttt_bpb-sw_bpb:+.4f})')
log0(f'After TTT + T=0.98:      {t_bpb:.4f} (delta: {t_bpb-sw_bpb:+.4f})')
log0(f'Total time: {time.perf_counter()-t0:.0f}s')

if distributed:
    dist.destroy_process_group()
