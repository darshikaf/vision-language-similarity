# Time and Memory Footprint Analysis: CLIP Similarity Computation

This document provides an analysis of the similarity metric's time and memory footprint, in response to the question:
> Briefly discuss the time and memory footprint of computing the similarity metric. What strategies could you use to optimise the time and/or memory performance of the code?

## Time Analysis

**Current Performance (ViT-B-32)**:
- **Single evaluation**: 172ms average per image
- **Batch evaluation**: 8.77s for 51 images (5.81 images/sec, 5.29x speedup)

**Key bottlenecks**:
1. **I/O dominance**: Image loading consumes majority of processing time
2. **Sequential tensor operations**: No pipelining of preprocessing steps
3. **Individual model calls**: Current batch processing uses concurrent individual inferences rather than true GPU batching

## Memory Footprint Analysis

**Dynamic memory per request**:
- **Concurrent batch scaling**: Linear (51 concurrent = ~2.5GB additional peak)

**Memory leak sources identified**:
- **ThreadPoolExecutor**: Not properly cleaned up
- **PyTorch tensors**: Potential accumulation despite `torch.no_grad()`

## Core Performance Limitations

The current architecture achieves good throughput through concurrency but has fundamental inefficiencies:

1. **False batching**: `asyncio.gather()` provides concurrency benefits but doesn't leverage GPU batch processing capabilities
2. **Memory churn**: Repeated tensor allocation/deallocation for each request
3. **Resource leaks**: Thread pools and tensor memory not properly managed

## Optimization Strategies

### Time Performance Optimizations

**1. True GPU Batch Processing**

**2. Async Preprocessing Pipeline**

Stream image loading, tensor conversion, and model inference in overlapping stages rather than sequential processing.

**3. Feature Caching**

Implement LRU cache for image feature vectors to handle repeated images.

**4. Mixed Precision Expansion**

Extend fp16 support beyond CUDA to MPS (Apple Silicon).

### Memory Performance Optimizations

**1. Memory Pool Management**

Implement tensor pooling to reuse allocated memory across requests

**2. Streaming Batch Processing**

Process large batches in memory-efficient chunks with explicit cleanup

**3. Resource Lifecycle Management**

Implement proper cleanup of thread pools and GPU memory

**4. Direct Tensor Image Loading**

Load images directly as tensors using torchvision.

### Architecture Optimizations

**1. Hybrid Processing Strategy**

Combine concurrent processing benefits with true GPU batching:
- Group requests into optimal batch sizes (8-16 images)
- Process multiple batches concurrently
- Balance memory usage with throughput

**2. Intelligent Batch Sizing**

Dynamically determine optimal batch size based on available GPU memory and model requirements.

