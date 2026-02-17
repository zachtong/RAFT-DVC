"""
Inference script for RAFT-DVC

Usage:
    python scripts/infer.py --checkpoint model.pth --vol0 ref.npy --vol1 def.npy --output flow.npy
    python scripts/infer.py --checkpoint model.pth --data_dir /path/to/data --output_dir /path/to/output
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from src.core import RAFTDVC


def parse_args():
    parser = argparse.ArgumentParser(description='Run RAFT-DVC inference')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Input (single pair or directory)
    parser.add_argument('--vol0', type=str, default=None,
                        help='Path to reference volume')
    parser.add_argument('--vol1', type=str, default=None,
                        help='Path to deformed volume')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing vol0/ and vol1/ subdirs')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for single pair inference')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory for batch inference')
    
    # Inference parameters
    parser.add_argument('--iters', type=int, default=24,
                        help='Number of refinement iterations')
    parser.add_argument('--patch_size', type=int, nargs=3, default=None,
                        help='Patch size for sliding window (H W D)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap ratio for sliding window')
    
    return parser.parse_args()


def load_volume(path: str) -> np.ndarray:
    """Load volume from file."""
    path = Path(path)
    ext = path.suffix.lower()
    
    if ext == '.npy':
        return np.load(str(path))
    elif ext in ['.tif', '.tiff']:
        import tifffile
        return tifffile.imread(str(path))
    elif ext in ['.h5', '.hdf5']:
        import h5py
        with h5py.File(str(path), 'r') as f:
            key = list(f.keys())[0]
            return f[key][:]
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_flow(flow: np.ndarray, path: str):
    """Save flow to file."""
    path = Path(path)
    ext = path.suffix.lower()
    
    if ext == '.npy':
        np.save(str(path), flow)
    elif ext in ['.h5', '.hdf5']:
        import h5py
        with h5py.File(str(path), 'w') as f:
            f.create_dataset('flow', data=flow, compression='gzip')
    else:
        # Default to npy
        np.save(str(path.with_suffix('.npy')), flow)


def infer_single_pair(
    model: RAFTDVC,
    vol0: np.ndarray,
    vol1: np.ndarray,
    device: torch.device,
    iters: int = 24
) -> tuple:
    """
    Run inference on a single volume pair.

    Returns:
        Tuple of (flow, uncertainty) where:
        - flow: (3, H, W, D) displacement field
        - uncertainty: (3, H, W, D) exp(log_b), or None if model
          has no uncertainty head
    """
    # Prepare input tensors
    vol0_t = torch.from_numpy(vol0.astype(np.float32))
    vol1_t = torch.from_numpy(vol1.astype(np.float32))

    # Add batch and channel dimensions if needed
    if vol0_t.ndim == 3:
        vol0_t = vol0_t.unsqueeze(0).unsqueeze(0)
        vol1_t = vol1_t.unsqueeze(0).unsqueeze(0)
    elif vol0_t.ndim == 4:
        vol0_t = vol0_t.unsqueeze(0)
        vol1_t = vol1_t.unsqueeze(0)

    vol0_t = vol0_t.to(device)
    vol1_t = vol1_t.to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(vol0_t, vol1_t, iters=iters, test_mode=True)

    # test_mode returns 2-tuple or 3-tuple (with uncertainty)
    flow = output[1].squeeze(0).cpu().numpy()

    uncertainty = None
    alpha = None
    if len(output) == 3:
        unc_raw = output[2]
        if unc_raw.shape[1] == 4:
            # MoL mode: channels 0-2 = log_b2, channel 3 = logit_alpha
            uncertainty = torch.exp(unc_raw[:, :3]).squeeze(0).cpu().numpy()
            alpha = torch.sigmoid(unc_raw[:, 3:4]).squeeze(0).cpu().numpy()
        else:
            # NLL mode: 3 channels = log_b
            uncertainty = torch.exp(unc_raw).squeeze(0).cpu().numpy()

    return flow, uncertainty, alpha


def infer_sliding_window(
    model: RAFTDVC,
    vol0: np.ndarray,
    vol1: np.ndarray,
    device: torch.device,
    patch_size: tuple,
    overlap: float = 0.5,
    iters: int = 24
) -> np.ndarray:
    """
    Run inference with sliding window for large volumes.
    
    Uses Gaussian weighting to blend overlapping patches.
    
    Returns:
        Flow field of shape (3, H, W, D)
    """
    H, W, D = vol0.shape[-3:]
    ph, pw, pd = patch_size
    
    # Calculate step size
    step_h = int(ph * (1 - overlap))
    step_w = int(pw * (1 - overlap))
    step_d = int(pd * (1 - overlap))
    
    # Create output arrays
    flow_sum = np.zeros((3, H, W, D), dtype=np.float32)
    weight_sum = np.zeros((1, H, W, D), dtype=np.float32)
    
    # Create Gaussian weight for blending
    sigma = min(patch_size) / 4
    weight = _create_gaussian_weight(patch_size, sigma)
    
    # Sliding window positions
    positions = []
    for h in range(0, H - ph + 1, step_h):
        for w in range(0, W - pw + 1, step_w):
            for d in range(0, D - pd + 1, step_d):
                positions.append((h, w, d))
    
    # Add edge cases
    if (H - ph) % step_h != 0:
        for w in range(0, W - pw + 1, step_w):
            for d in range(0, D - pd + 1, step_d):
                positions.append((H - ph, w, d))
    
    model.eval()
    with torch.no_grad():
        for h, w, d in tqdm(positions, desc="Inferring patches"):
            # Extract patch
            vol0_patch = vol0[..., h:h+ph, w:w+pw, d:d+pd]
            vol1_patch = vol1[..., h:h+ph, w:w+pw, d:d+pd]
            
            # Run inference
            flow_patch, _, _ = infer_single_pair(
                model, vol0_patch, vol1_patch, device, iters
            )
            
            # Accumulate with weighting
            flow_sum[:, h:h+ph, w:w+pw, d:d+pd] += flow_patch * weight
            weight_sum[:, h:h+ph, w:w+pw, d:d+pd] += weight
    
    # Normalize
    flow = flow_sum / (weight_sum + 1e-8)
    
    return flow


def _create_gaussian_weight(shape: tuple, sigma: float) -> np.ndarray:
    """Create 3D Gaussian weight for patch blending."""
    h, w, d = shape
    
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    z = np.linspace(-1, 1, d)
    
    yy, xx, zz = np.meshgrid(y, x, z, indexing='ij')
    
    weight = np.exp(-(yy**2 + xx**2 + zz**2) / (2 * (sigma/min(shape))**2))
    
    return weight.astype(np.float32)


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, _ = RAFTDVC.load_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"Model loaded with {model.get_num_parameters():,} parameters")
    
    # Single pair inference
    if args.vol0 is not None and args.vol1 is not None:
        print(f"Loading volumes...")
        vol0 = load_volume(args.vol0)
        vol1 = load_volume(args.vol1)
        print(f"Volume shape: {vol0.shape}")
        
        if args.patch_size is not None:
            print(f"Using sliding window with patch size {args.patch_size}")
            flow = infer_sliding_window(
                model, vol0, vol1, device,
                patch_size=tuple(args.patch_size),
                overlap=args.overlap,
                iters=args.iters
            )
            uncertainty = None  # sliding window doesn't support uncertainty yet
            alpha = None
        else:
            print("Running single-pass inference...")
            flow, uncertainty, alpha = infer_single_pair(model, vol0, vol1, device, args.iters)

        # Save result
        output_path = args.output or 'flow_result.npy'
        save_flow(flow, output_path)
        print(f"Flow saved to {output_path}")
        print(f"Flow shape: {flow.shape}")
        print(f"Flow magnitude range: [{np.linalg.norm(flow, axis=0).min():.3f}, "
              f"{np.linalg.norm(flow, axis=0).max():.3f}]")

        # Save uncertainty if available
        if uncertainty is not None:
            uncert_path = Path(output_path).with_name(
                Path(output_path).stem + '_uncertainty.npy'
            )
            np.save(str(uncert_path), uncertainty)
            print(f"Uncertainty saved to {uncert_path}")
            print(f"Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")

        # Save alpha (confidence) if MoL mode
        if alpha is not None:
            alpha_path = Path(output_path).with_name(
                Path(output_path).stem + '_alpha.npy'
            )
            np.save(str(alpha_path), alpha)
            print(f"MoL alpha (confidence) saved to {alpha_path}")
            print(f"Alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
    
    # Batch inference
    elif args.data_dir is not None:
        data_dir = Path(args.data_dir)
        vol0_dir = data_dir / 'vol0'
        vol1_dir = data_dir / 'vol1'
        
        if not vol0_dir.exists() or not vol1_dir.exists():
            print(f"Error: Expected vol0/ and vol1/ subdirectories in {data_dir}")
            return
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = sorted(vol0_dir.glob('*.npy'))
        print(f"Found {len(files)} volume pairs")
        
        for vol0_path in files:
            vol1_path = vol1_dir / vol0_path.name
            if not vol1_path.exists():
                print(f"Warning: No matching vol1 for {vol0_path.name}")
                continue
            
            print(f"Processing {vol0_path.name}...")
            vol0 = load_volume(str(vol0_path))
            vol1 = load_volume(str(vol1_path))
            
            if args.patch_size is not None:
                flow = infer_sliding_window(
                    model, vol0, vol1, device,
                    patch_size=tuple(args.patch_size),
                    overlap=args.overlap,
                    iters=args.iters
                )
                uncertainty = None
                alpha = None
            else:
                flow, uncertainty, alpha = infer_single_pair(
                    model, vol0, vol1, device, args.iters
                )

            output_path = output_dir / f"flow_{vol0_path.stem}.npy"
            save_flow(flow, str(output_path))
            print(f"  Saved to {output_path}")

            if uncertainty is not None:
                uncert_path = output_dir / f"uncertainty_{vol0_path.stem}.npy"
                np.save(str(uncert_path), uncertainty)
                print(f"  Uncertainty saved to {uncert_path}")

            if alpha is not None:
                alpha_path = output_dir / f"alpha_{vol0_path.stem}.npy"
                np.save(str(alpha_path), alpha)
                print(f"  MoL alpha saved to {alpha_path}")
        
        print(f"Batch inference completed. Results in {output_dir}")
    
    else:
        print("Error: Must specify either --vol0/--vol1 or --data_dir")


if __name__ == '__main__':
    main()
