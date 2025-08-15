# Implementation Details

## Model Loading Strategy: Runtime vs Pre-baked

**Runtime model loading** was selected over pre-baking models into Docker images to enable dynamic model management in production. While pre-baking offers faster startup times (~1s vs 6s), runtime loading provides flexibility for A/B testing, emergency model rollbacks, and separate model release cycles from application deployments. Alternative approaches include hybrid strategies (critical models pre-baked, others runtime), persistent volume mounting, or init container patterns, but these add operational complexity without addressing the need for hot-swappable model configurations.

## Kubernetes ConfigMap Integration

The implementation uses Kubernetes ConfigMaps loaded as environment variables, allowing the `DynamicModelRegistry` to dynamically reconfigure model specifications (model architecture, pretrained weights, performance characteristics) without code changes. To safely update models without redeployments, **Reloader** can be employed - a Kubernetes controller that automatically triggers rolling updates when ConfigMaps change. This ensures all pods restart gracefully with new environment variables, maintaining zero-downtime deployments while enabling real-time model configuration updates.

This architecture separates model configuration from application logic, enabling data scientists to update model specifications independently of engineering deployments while maintaining production stability through Kubernetes' native rolling update mechanisms.

## Pluggable Model Architecture

The application implements a plugin-based similarity model system through the `SimilarityModel` abstract base class and `SimilarityModelFactory` registry pattern. This design allows integration of different vision-language model backends (currently OpenCLIP, but extensible to HuggingFace Transformers CLIP models, direct PyTorch CLIP implementations, or custom vision-language architectures) without modifying core evaluation logic. The factory pattern combined with runtime model loading enables not only hot-swapping between different configurations of the same model type, but also switching between entirely different model architectures through ConfigMap updates, providing flexibility for model experimentation and production optimization.

## Async-First Architecture with ML Optimizations

The service employs asynchronous patterns specifically designed for ML workloads. PyTorch operations are isolated to dedicated thread pools to prevent blocking the async event loop, while image loading leverages `asyncio.gather()` for parallel processing with connection pooling.

## Production-Grade Observability

The observability stack provides ML-specific monitoring through custom Prometheus metrics including CLIP score distributions, model inference timing histograms, and batch efficiency ratios. System resource tracking covers CPU, memory, and GPU utilization with automatic updates, while structured error categorization enables issue identification. In the future, OpenTelemetry integration provides histogram buckets to optimize monitoring for ML workload characteristics. As a next step, distributed tracing for request flows should be implemented.

## Future Similarity Score Enhancements

The pluggable model architecture enables similarity scoring improvements. **Multi-Metric Score Fusion** can be implemented within the `SimilarityModel` interface by extending `compute_similarity()` to return multiple scores (VQAScore, TIFA, compositional alignment) that are weighted and combined. **Adaptive Image Preprocessing** leverages the existing preprocessing pipeline to implement domain-specific transformations based on prompt analysis, while **Score Calibration** can be added as a configurable component in the model factory pattern. These enhancements integrate seamlessly with the ConfigMap-driven configuration system, enabling data scientists to experiment with different scoring approaches without code changes, from simple CLIP similarity to sophisticated VQA-based compositional understanding.
