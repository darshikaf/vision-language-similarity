# Architecture & Design Decisions

## Model Loading Strategy: Runtime vs Pre-baked

**Runtime model loading** was selected over pre-baking models into Docker images to enable dynamic model management in production. While pre-baking offers faster startup times (~1s vs 6s), runtime loading provides flexibility for A/B testing, emergency model rollbacks, and separate model release cycles from application deployments. Alternative approaches include hybrid strategies (critical models pre-baked, others runtime), persistent volume mounting, or init container patterns, but these add operational complexity without addressing the need for hot-swappable model configurations.

## Config Driven Dynamic model registry

The implementation uses a config driven approach, allowing the `DynamicModelRegistry` to dynamically reconfigure model specifications (model architecture, pretrained weights, performance characteristics) without code changes.

## Pluggable Model Architecture

The application implements a plugin-based similarity model system through the `SimilarityModel` abstract base class and `SimilarityModelFactory` registry pattern. This design allows integration of different vision-language model backends (currently OpenCLIP, but extensible to HuggingFace Transformers CLIP models, direct PyTorch CLIP implementations, or custom vision-language architectures) without modifying core evaluation logic. The factory pattern combined with runtime model loading enables not only hot-swapping between different configurations of the same model type, but also switching between entirely different model architectures through ConfigMap updates, providing flexibility for model experimentation and production optimization.

```
Configuration Layer Changes

  1. Create generic ModelSpec base class in service/core/ml/utils/config.py:12-20 to replace hardcoded CLIPModelSpec
  2. Add new model configuration to DynamicModelRegistry.DEFAULT_MODELS in service/core/ml/utils/config.py:31-42
  3. Update factory config extraction logic in service/core/ml/models/factory.py:32-51 to handle multiple model types generically

Model Layer Changes

  4. Register new model class in SimilarityModelFactory._MODEL_REGISTRY in service/core/ml/models/factory.py:12-14
  5. Add new model import to service/core/ml/models/factory.py:5
  6. Create new SimilarityModel implementation (e.g., service/core/ml/models/huggingface_model.py)

Evaluator Layer Changes

  7. Create EvaluatorFactory class (new file: service/core/ml/engines/evaluator_factory.py)
  8. Create new Evaluator implementation (e.g., service/core/ml/engines/huggingface_evaluator.py)

Handler Layer Changes

  9. Change evaluator type annotation from OpenCLIPEvaluator to AbstractEvaluator in service/evaluation/handler.py:25
  10. Replace hardcoded evaluator creation with factory pattern in service/evaluation/handler.py:34
  11. Update handler imports to remove OpenCLIPEvaluator and add EvaluatorFactory in service/evaluation/handler.py:4
  12. Update health check logic to use abstract interface instead of OpenCLIP-specific attributes in service/evaluation/handler.py:136-141

Core Module Changes

  13. Update core module exports in service/core/__init__.py to remove hardcoded OpenCLIPEvaluator import

Configuration File Support (Optional)

  14. Create model configuration template in /app/config/models.json for new model types
  15. Add environment variable support for new model configurations (already supported by existing DynamicModelRegistry)
```

## Future Similarity Score Enhancements

The pluggable model architecture enables similarity scoring improvements. **Multi-Metric Score Fusion** can be implemented within the `SimilarityModel` interface by extending `compute_similarity()` to return multiple scores (VQAScore (Visual Question Answering), TIFA (Text-to-Image Faithfulness), compositional alignment) that are weighted and combined. **Adaptive Image Preprocessing** can leverage the existing preprocessing pipeline to implement domain-specific transformations based on prompt analysis, while **Score Calibration** can be added as a configurable component in the model factory pattern. These enhancements integrate with the ConfigMap-driven configuration system, enabling experiment with different scoring approaches without code changes, from simple CLIP similarity to sophisticated VQA-based compositional understanding.

* Multi-metric score fusion
    * currently returns `max(100 * cosine_similarity, 0)`
    * VQAScore can verify spatial relationships that CLIP might miss
    * TIFA can add fine-grained attribute verification
    * compositional alignment can add object-relation-object relations
* Adaptive image processing
    * currently basic RGB conversion + openCLIP's default preprocessing
    * semantic aware preprocessing
        * edge enhancement for "detailed architecture" prompts
        * color saturation for "vibrant" descriptions
* Score calibration
    * Different model configs can have different score distributions
    * Use temperature scaling to calibrate confidence using some validation data

