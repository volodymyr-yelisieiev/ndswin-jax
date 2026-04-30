"""Inference utilities for NDSwin-JAX."""

from ndswin.inference.batch_processor import (
    BatchProcessor,
    process_dataset,
)
from ndswin.inference.export import (
    export_for_serving,
    export_to_onnx,
    export_to_saved_model,
)
from ndswin.inference.predictor import (
    ClassificationPredictor,
    Predictor,
)

__all__ = [
    # Predictor
    "Predictor",
    "ClassificationPredictor",
    # Batch processing
    "BatchProcessor",
    "process_dataset",
    # Export
    "export_to_onnx",
    "export_to_saved_model",
    "export_for_serving",
]
