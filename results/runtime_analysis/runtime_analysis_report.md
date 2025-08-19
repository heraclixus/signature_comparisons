# Training Runtime Analysis Report
Generated on: 2025-08-19 09:23:53

## System Information
- CPU: 12 cores @ 4.0 MHz
- Memory: 24.0 GB
- CUDA: Not Available
- PyTorch: 2.8.0

## Training Speed Analysis

### Training Speed Ranking (Fastest to Slowest):

1. **B5**: 0.28s per epoch
2. **A1**: 1.13s per epoch
3. **A4**: 1.16s per epoch
4. **B1**: 26.33s per epoch
5. **B2**: 47.89s per epoch

### Training Efficiency Ranking (Best Loss Improvement Rate):

1. **B5**: 0.0416 loss improvement per second
2. **A1**: 0.0176 loss improvement per second
3. **A4**: 0.0072 loss improvement per second
4. **B1**: 0.0013 loss improvement per second
5. **B2**: 0.0002 loss improvement per second

### Model Complexity vs Performance:

| Model | Parameters | Avg Epoch Time | Time to Convergence |
|-------|------------|----------------|---------------------|
| A1 | 199.0 | 1.13s | 5.7s |
| A4 | 199.0 | 1.16s | 5.8s |
| B1 | 9,027.0 | 26.33s | 131.7s |
| B2 | 9,027.0 | 47.89s | 191.6s |
| B5 | 9,027.0 | 0.28s | 1.4s |

## Detailed Performance Profiling

### Training Phase Breakdown:

**A1**:
- Forward pass: 28.5%
- Loss computation: 2.5%
- Backward pass: 68.7%
- Throughput: 215.9 samples/sec
- Memory usage: 4.9 MB per step

**A2**:
- Forward pass: 30.2%
- Loss computation: 0.9%
- Backward pass: 68.6%
- Throughput: 232.8 samples/sec
- Memory usage: 1.2 MB per step

**A3**:
- Forward pass: 27.6%
- Loss computation: 7.1%
- Backward pass: 65.1%
- Throughput: 215.0 samples/sec
- Memory usage: 10.7 MB per step

**A4**:
- Forward pass: 29.6%
- Loss computation: 2.6%
- Backward pass: 67.6%
- Throughput: 231.1 samples/sec
- Memory usage: 0.2 MB per step

**B2**:
- Forward pass: 1.0%
- Loss computation: 97.8%
- Backward pass: 1.2%
- Throughput: 10.8 samples/sec
- Memory usage: 64.9 MB per step

**B3**:
- Forward pass: 41.2%
- Loss computation: 5.6%
- Backward pass: 52.7%
- Throughput: 462.2 samples/sec
- Memory usage: 0.5 MB per step

**B4**:
- Forward pass: 42.8%
- Loss computation: 2.8%
- Backward pass: 54.0%
- Throughput: 488.1 samples/sec
- Memory usage: 0.2 MB per step

**B5**:
- Forward pass: 42.3%
- Loss computation: 2.2%
- Backward pass: 55.2%
- Throughput: 485.2 samples/sec
- Memory usage: -0.0 MB per step


## Recommendations

### For Speed-Critical Applications:
- **Fastest Training**: B5 (0.28s per epoch)

### For Efficiency-Critical Applications:
- **Best Efficiency**: B5 (0.0416 improvement rate)

### Architecture Comparison:
- **CannedNet models**: 1.14s average epoch time
- **Neural SDE models**: 24.83s average epoch time
- **Speed ratio**: Neural SDE is 21.7x slower than CannedNet
