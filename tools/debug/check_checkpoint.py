"""Quick script to check checkpoint contents."""
import torch
import sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/training/confocal_baseline/checkpoint_epoch_100.pth'

print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path)

print("\n=== Checkpoint Contents ===")
print(f"Keys: {list(ckpt.keys())}")
print(f"\nEpoch: {ckpt.get('epoch', 'N/A')}")
print(f"Best EPE: {ckpt.get('best_epe', 'N/A')}")
print(f"Has scheduler_state_dict: {'scheduler_state_dict' in ckpt}")

if 'scheduler_state_dict' in ckpt:
    print("\n=== Scheduler State ===")
    sched = ckpt['scheduler_state_dict']
    print(f"Scheduler keys: {list(sched.keys())}")
    if 'last_epoch' in sched:
        print(f"Last epoch (iteration): {sched['last_epoch']}")
    if '_step_count' in sched:
        print(f"Step count: {sched['_step_count']}")
else:
    print("\n⚠️ WARNING: Checkpoint does NOT contain scheduler state!")
    print("   Scheduler will restart from step 0, causing LR reset.")
