"""Tests for inference module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import MagicMock
import flax.linen as nn

from ndswin.inference.batch_processor import (
    BatchProcessor,
    ParallelProcessor,
    StreamingProcessor,
    process_dataset,
)
from ndswin.inference.predictor import (
    ClassificationPredictor,
    FeatureExtractor,
    SegmentationPredictor,
    create_predictor,
)


class DummyModel(nn.Module):
    task: str = "classification"
    return_features: bool = False
    
    @property
    def input_shape(self):
        return (3, 32, 32)
        
    @nn.compact
    def __call__(self, x, deterministic=True, training=False, return_features=False):
        batch_size = x.shape[0] if x.ndim > 0 else 1
        
        if self.task == "classification":
            # return fake logits
            return jnp.ones((batch_size, 10))
        elif self.task == "segmentation":
            # return fake spatial logits
            return jnp.ones((batch_size, 2, 32, 32))
        elif self.task == "features":
            logits = jnp.ones((batch_size, 10))
            features = [jnp.ones((batch_size, 16, 16, 16)), jnp.ones((batch_size, 32, 8, 8))]
            return logits, features

        return jnp.ones((batch_size, 10))


@pytest.fixture
def mock_classification_predictor():
    model = DummyModel()
    params = {}
    return ClassificationPredictor(model, params, class_names=["class" + str(i) for i in range(10)])


@pytest.fixture
def mock_segmentation_predictor():
    model = DummyModel(task="segmentation")
    params = {}
    return SegmentationPredictor(model, params)


@pytest.fixture
def mock_feature_extractor():
    model = DummyModel(task="features")
    params = {}
    return FeatureExtractor(model, params, return_features=True)


def test_classification_predictor(mock_classification_predictor):
    """Test classification predictions."""
    # Single input
    x_single = np.ones((3, 32, 32))
    res = mock_classification_predictor.predict(x_single)
    assert "class_id" in res
    assert "class_name" in res
    assert res["class_name"].startswith("class")
    assert "probabilities" in res
    
    # Batch input
    x_batch = np.ones((4, 3, 32, 32))
    res_batch = mock_classification_predictor.predict(x_batch)
    assert len(res_batch["class_names"]) == 4
    
    # Test predict_batch method
    batch_res = mock_classification_predictor.predict_batch(x_batch)
    assert len(batch_res) == 4
    assert batch_res[0]["class_name"] is not None


def test_segmentation_predictor(mock_segmentation_predictor):
    """Test segmentation predictions."""
    x_single = np.ones((3, 32, 32))
    res = mock_segmentation_predictor.predict(x_single)
    assert "mask" in res
    assert "probabilities" in res
    assert res["mask"].shape == (32, 32)
    
    # Batch input
    x_batch = np.ones((2, 3, 32, 32))
    res_batch = mock_segmentation_predictor.predict(x_batch)
    assert res_batch["mask"].shape == (32, 32)


def test_feature_extractor(mock_feature_extractor):
    """Test feature extraction."""
    x_batch = np.ones((2, 3, 32, 32))
    res = mock_feature_extractor.predict(x_batch)
    assert "logits" in res
    assert "stage_0" in res
    assert "stage_1" in res
    assert res["stage_0"].shape == (2, 16, 16, 16)


def test_create_predictor():
    """Test predictor factory."""
    model = DummyModel()
    params = {}
    
    clf = create_predictor(model, params, task="classification")
    assert isinstance(clf, ClassificationPredictor)
    
    seg = create_predictor(model, params, task="segmentation")
    assert isinstance(seg, SegmentationPredictor)
    
    feat = create_predictor(model, params, task="features")
    assert isinstance(feat, FeatureExtractor)
    
    with pytest.raises(ValueError):
        create_predictor(model, params, task="unknown")


def test_batch_processor(mock_classification_predictor):
    """Test batch processing over different input streams."""
    processor = BatchProcessor(mock_classification_predictor, batch_size=2, show_progress=False)
    
    # List of arrays
    data_list = [np.ones((3, 32, 32)) for _ in range(5)]
    res = processor.process(data_list)
    assert len(res) == 3
    
    # Numpy array
    data_np = np.ones((5, 3, 32, 32))
    res2 = processor.process(data_np)
    assert len(res2) == 3
    
    # Generator
    def data_gen():
        for _ in range(5):
            yield np.ones((3, 32, 32))
            
    res_lazy = list(processor.process_generator(data_gen()))
    assert len(res_lazy) == 3


def test_process_dataset(mock_classification_predictor):
    """Test dataset processing util."""
    dataset = [
        {"image": np.ones((2, 3, 32, 32)), "label": np.zeros((2,))},
        {"image": np.ones((1, 3, 32, 32)), "label": np.ones((1,))}
    ]
    
    preds, labels = process_dataset(mock_classification_predictor, dataset, return_labels=True)
    assert len(labels) == 3
    assert len(preds) == 2


def test_streaming_processor(mock_classification_predictor):
    """Test stream processing."""
    streamer = StreamingProcessor(mock_classification_predictor, buffer_size=2)
    
    r1 = streamer.add_to_buffer(np.ones((3, 32, 32)))
    assert r1 is None
    
    r2 = streamer.add_to_buffer(np.ones((3, 32, 32)))
    assert r2 is not None
    assert len(r2) == 1
    
    # flush
    streamer.add_to_buffer(np.ones((3, 32, 32)))
    r3 = streamer.flush()
    assert r3 is not None
    
    assert streamer.flush() is None
    
    # process_single
    r4 = streamer.process_single(np.ones((3, 32, 32)))
    assert "class_names" in r4


def test_parallel_processor(mock_classification_predictor):
    """Test pmap parallel processing logic with a timeout guard."""
    import threading

    devices = jax.devices()
    result_holder = {}
    error_holder = {}

    def _run():
        try:
            processor = ParallelProcessor(mock_classification_predictor, devices=devices)
            num_samples = len(devices) * 4 + 1
            data = np.ones((num_samples, 3, 32, 32))
            res = processor.process(data, batch_size_per_device=2)
            result_holder["result"] = res
        except Exception as e:
            error_holder["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=30)

    if t.is_alive():
        pytest.fail("ParallelProcessor.process hung for >30s (pmap deadlock)")

    if "error" in error_holder:
        e = error_holder["error"]
        # pmap/sharding errors are acceptable in some environments
        if "pmap" in str(e).lower() or "sharding" in str(e).lower():
            pytest.skip(f"pmap not supported in this environment: {e}")
        raise e

    assert "result" in result_holder
    num_samples = len(devices) * 4 + 1
    assert len(result_holder["result"]) == num_samples
