"""Quick CLI verification: save a dummy weight then run test.py on tiny data."""
import subprocess, sys, os, torch, numpy as np, h5py

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 0) Create a tiny h5 file with 2 samples
print("=== Creating tiny test h5 ===")
os.makedirs("data/wv3", exist_ok=True)
with h5py.File("data/wv3/_tiny_test.h5", "w") as f:
    f.create_dataset("gt", data=np.random.rand(2, 8, 64, 64).astype(np.float32) * 2047)
    f.create_dataset("lms", data=np.random.rand(2, 8, 64, 64).astype(np.float32) * 2047)
    f.create_dataset("ms", data=np.random.rand(2, 8, 16, 16).astype(np.float32) * 2047)
    f.create_dataset("pan", data=np.random.rand(2, 1, 64, 64).astype(np.float32) * 2047)
print("Created data/wv3/_tiny_test.h5")

# 1) Create a dummy weight file
print("=== Creating dummy weights ===")
from models.dit_pan import DiTPan_S
from diffusion.shift_diffusion import ShiftDiffusion, make_sqrt_etas_schedule

model = DiTPan_S(input_size=64, in_channels=8, lms_channel=8, pan_channel=1, self_condition=True)
diff = ShiftDiffusion(model, channels=8, pred_mode="x_start", loss_type="l2", device="cpu", clamp_range=(0,1))
diff.set_new_noise_schedule(sqrt_etas=make_sqrt_etas_schedule("cosine", 2), device="cpu")

os.makedirs("weights", exist_ok=True)
# Save only the inner model state dict (same as ema_model_state_dict in train.py)
torch.save(model.state_dict(), "weights/_test_dummy.pth")
print("Saved weights/_test_dummy.pth")

# 2) Run test.py on the tiny data
print("\n=== Running test.py ===")
result = subprocess.run(
    [sys.executable, "test.py",
     "--test_path", "data/wv3/_tiny_test.h5",
     "--weight_path", "weights/_test_dummy.pth",
     "--dataset_name", "wv3",
     "--device", "cpu",
     "--n_steps", "2",
     "--batch_size", "2",
     "--image_size", "64",
    ],
    capture_output=True, text=True, timeout=600,
)
print("STDOUT:", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    print(f"[FAIL] test.py exited with code {result.returncode}")
    sys.exit(1)
else:
    print("[PASS] test.py completed successfully")

# Cleanup
os.remove("weights/_test_dummy.pth")
os.remove("data/wv3/_tiny_test.h5")
for f in os.listdir("samples/mat"):
    if "DiTPan" in f:
        os.remove(os.path.join("samples/mat", f))
print("Cleaned up test artifacts")
