"""
测试CUDA是否真正可用
"""
import torch
import torch.nn as nn

print("="*60)
print("CUDA Functionality Test")
print("="*60)

# 基本信息
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 测试1: 创建tensor
print("\n[Test 1] Creating tensor on GPU...")
try:
    x = torch.randn(100, 100).cuda()
    print("✓ Tensor creation: SUCCESS")
except Exception as e:
    print(f"✗ Tensor creation: FAILED - {e}")
    exit(1)

# 测试2: 矩阵运算
print("\n[Test 2] Matrix multiplication on GPU...")
try:
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    c = torch.matmul(a, b)
    print(f"✓ Matrix multiplication: SUCCESS")
    print(f"  Result shape: {c.shape}")
except Exception as e:
    print(f"✗ Matrix multiplication: FAILED - {e}")
    exit(1)

# 测试3: 卷积操作
print("\n[Test 3] 3D convolution on GPU...")
try:
    conv = nn.Conv3d(1, 8, kernel_size=3, padding=1).cuda()
    input_vol = torch.randn(1, 1, 32, 32, 32).cuda()
    output = conv(input_vol)
    print(f"✓ 3D convolution: SUCCESS")
    print(f"  Input shape: {input_vol.shape}")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"✗ 3D convolution: FAILED - {e}")
    exit(1)

# 测试4: 反向传播
print("\n[Test 4] Backward pass on GPU...")
try:
    x = torch.randn(10, 10, requires_grad=True).cuda()
    y = (x ** 2).sum()
    y.backward()
    print(f"✓ Backward pass: SUCCESS")
    print(f"  Gradient shape: {x.grad.shape}")
except Exception as e:
    print(f"✗ Backward pass: FAILED - {e}")
    exit(1)

print("\n" + "="*60)
print("All tests PASSED! ✓")
print("RTX 5090 is fully functional with PyTorch!")
print("="*60)
