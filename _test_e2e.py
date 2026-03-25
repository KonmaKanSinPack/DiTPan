"""End-to-end integration test for DiTPan pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import h5py
import einops
import torch.nn.functional as F
from copy import deepcopy
from models.dit_pan import DiTPan_S, interpolate_pos_embed
from diffusion.shift_diffusion import ShiftDiffusion, make_sqrt_etas_schedule
from dataset.pan_dataset import PanDataset
from utils.metric import AnalysisPanAcc
from utils.optim_utils import EmaUpdater
from utils.misc import grad_clip, model_load
from utils.lr_scheduler import StepsAll

device = 'cpu'
os.makedirs('./logs', exist_ok=True)
os.makedirs('./runs', exist_ok=True)
os.makedirs('./weights', exist_ok=True)

print('=== 1. Position interpolation ===')
pos = torch.randn(1, 1024, 384)
new_pos = interpolate_pos_embed(pos, 4096)
assert new_pos.shape == (1, 4096, 384)
same = interpolate_pos_embed(pos, 1024)
assert torch.equal(same, pos)
print('[PASS]')

print('=== 2. Dataset loading ===')
d = h5py.File('data/wv3/train_wv3.h5', 'r')
ds = PanDataset(d, full_res=False, norm_range=False, division=2047.0, aug_prob=0, wavelets=True)
print(f'Dataset: {len(ds)} samples')
pan, lms, gt, wavelets = ds[0]
pan, lms, gt, wavelets = [x.unsqueeze(0) for x in [pan, lms, gt, wavelets]]
print(f'pan:{pan.shape} lms:{lms.shape} gt:{gt.shape} wav:{wavelets.shape}')

print('=== 3. Condition packing ===')
cond, _ = einops.pack(
    [lms, pan, F.interpolate(wavelets, size=lms.shape[-1], mode='bilinear')],
    'b * h w',
)
print(f'cond: {cond.shape}')
assert cond.shape == (1, 20, 64, 64), f'Wrong: {cond.shape}'
print('[PASS]')

print('=== 4. Model + Diffusion ===')
model = DiTPan_S(input_size=64, in_channels=8, lms_channel=8, pan_channel=1, self_condition=True).to(device)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Params: {n_params:.2f}M')

sqrt_etas = make_sqrt_etas_schedule('cosine', n_timestep=15)
diffusion = ShiftDiffusion(model, channels=8, pred_mode='x_start', loss_type='l2',
                          device=device, clamp_range=(0,1), penalty_weight=100.0)
diffusion.set_new_noise_schedule(sqrt_etas=sqrt_etas, device=device)
diffusion = diffusion.to(device)
print('[PASS]')

print('=== 5. Training step ===')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=0.2)
schedulers = StepsAll(scheduler)
ema_updater = EmaUpdater(diffusion, deepcopy(diffusion), decay=0.995, start_iter=20000)

optimizer.zero_grad()
loss = diffusion(x=gt, y=lms, cond=cond, mode='train', current_iter=0)
loss.backward()
grad_clip(model.parameters(), mode='norm', value=0.5)
optimizer.step()
ema_updater.update(1)
schedulers.step()
print(f'Loss: {loss.item():.6f} [PASS]')

print('=== 6. Metrics ===')
model.eval()
analysis = AnalysisPanAcc()
with torch.no_grad():
    t = torch.zeros(1, dtype=torch.long)
    out = model(lms, t, cond)
sr = (out + lms).clip(0, 1)
analysis(gt, sr)
print(f'Metrics: {analysis.print_str()} [PASS]')

print('=== 7. Sampling (2 steps) ===')
sqrt_etas2 = make_sqrt_etas_schedule('cosine', n_timestep=2)
diffusion.set_new_noise_schedule(sqrt_etas=sqrt_etas2, device=device)
with torch.no_grad():
    e0 = diffusion(y=lms, cond=cond, mode='ddpm_sample')
sr2 = (e0 + lms).clip(0, 1)
print(f'Sample: {e0.shape}, SR range: [{sr2.min():.3f}, {sr2.max():.3f}] [PASS]')

print('=== 8. Save/Load ===')
save_path = './weights/test_ema.pth'
torch.save(ema_updater.ema_model_state_dict, save_path)
model2 = DiTPan_S(input_size=64, in_channels=8, lms_channel=8, pan_channel=1, self_condition=True)
model2 = model_load(save_path, model2, device=device)
with torch.no_grad():
    out2 = model2(lms, t, cond)
print(f'Loaded model output: {out2.shape} [PASS]')

print('=== 9. Variable-size (128x128) ===')
x128 = torch.randn(1, 8, 128, 128)
cond128 = torch.randn(1, 20, 128, 128)
with torch.no_grad():
    out128 = model(x128, t, cond128)
assert out128.shape == (1, 8, 128, 128), f'Wrong: {out128.shape}'
print(f'128x128: {out128.shape} [PASS]')

# Cleanup test weights
os.remove(save_path)

print('\n=== ALL INTEGRATION TESTS PASSED ===')
