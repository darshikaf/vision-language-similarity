# Architecture & Design Decisions

## Model Loading Strategy: Runtime vs Pre-baked

**Runtime model loading** was selected over pre-baking models into Docker images to enable dynamic model management in production. While pre-baking offers faster startup times (~1s vs 6s), runtime loading provides flexibility for A/B testing, emergency model rollbacks, and separate model release cycles from application deployments. Alternative approaches include hybrid strategies (critical models pre-baked, others runtime), persistent volume mounting, or init container patterns, but these add operational complexity without addressing the need for hot-swappable model configurations.

## Kubernetes ConfigMap Integration

The implementation uses Kubernetes ConfigMaps loaded as environment variables, allowing the `DynamicModelRegistry` to dynamically reconfigure model specifications (model architecture, pretrained weights, performance characteristics) without code changes. To safely update models without redeployments, **Reloader** can be employed - a Kubernetes controller that automatically triggers rolling updates when ConfigMaps change. This ensures all pods restart gracefully with new environment variables, maintaining zero-downtime deployments while enabling real-time model configuration updates.

## Pluggable Model Architecture

The application implements a plugin-based similarity model system through the `SimilarityModel` abstract base class and `SimilarityModelFactory` registry pattern. This design allows integration of different vision-language model backends (currently OpenCLIP, but extensible to HuggingFace Transformers CLIP models, direct PyTorch CLIP implementations, or custom vision-language architectures) without modifying core evaluation logic. The factory pattern combined with runtime model loading enables not only hot-swapping between different configurations of the same model type, but also switching between entirely different model architectures through ConfigMap updates, providing flexibility for model experimentation and production optimization.

## Future Similarity Score Enhancements

The pluggable model architecture enables similarity scoring improvements. **Multi-Metric Score Fusion** can be implemented within the `SimilarityModel` interface by extending `compute_similarity()` to return multiple scores (VQAScore, TIFA, compositional alignment) that are weighted and combined. **Adaptive Image Preprocessing** can leverage the existing preprocessing pipeline to implement domain-specific transformations based on prompt analysis, while **Score Calibration** can be added as a configurable component in the model factory pattern. These enhancements integrate with the ConfigMap-driven configuration system, enabling experiment with different scoring approaches without code changes, from simple CLIP similarity to sophisticated VQA-based compositional understanding.
