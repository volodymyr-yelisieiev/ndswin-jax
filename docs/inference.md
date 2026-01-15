# Inference Guide

This guide covers how to use NDSwin-JAX models for inference.

## Loading a Model Checkpoint

```python
import jax
import jax.numpy as jnp
from ndswin import NDSwinConfig, NDSwinTransformer
from ndswin.models.pretrained import load_checkpoint

# Load checkpoint (assuming you have trained a model)
checkpoint = load_checkpoint("checkpoints/", step=100)

# Recreate model with same config
config = checkpoint.get('config', NDSwinConfig.swin_tiny_2d())
model = NDSwinTransformer(config=config)

# Get parameters
params = checkpoint['params']
variables = {'params': params}
```

## Basic Prediction

```python
# Prepare input
image = jnp.ones((1, 3, 224, 224))  # (batch, channels, height, width)

# Forward pass
logits = model.apply(variables, image, training=False)

# Get prediction
probs = jax.nn.softmax(logits)
predicted_class = jnp.argmax(probs, axis=-1)
confidence = probs[0, predicted_class[0]]

print(f"Predicted class: {predicted_class[0]}")
print(f"Confidence: {confidence:.2%}")
```

## Using ClassificationPredictor

High-level predictor with preprocessing and post-processing:

```python
from ndswin.inference import ClassificationPredictor

# Create predictor
predictor = ClassificationPredictor(
    model=model,
    params=params,
    class_names=["cat", "dog", "bird", ...],  # Optional
    preprocessing={"mean": (0.485, 0.456, 0.406), 
                   "std": (0.229, 0.224, 0.225)},
)

# Single prediction
result = predictor.predict(image)
print(f"Class: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top-5: {result['top_k']}")

# Batch prediction
results = predictor.predict_batch(images)
```

## Batch Processing

For processing large datasets efficiently:

```python
from ndswin.inference import BatchProcessor

# Create processor
processor = BatchProcessor(
    predictor=predictor,
    batch_size=32,
    num_workers=4,
)

# Process dataset
results = processor.process(images)

# Or from files
results = processor.process_files(image_paths)

# With progress bar
for batch_results in processor.process_iter(images, progress=True):
    # Handle batch results
    pass
```

## Feature Extraction

Extract features for downstream tasks:

```python
from ndswin.inference import FeatureExtractor

extractor = FeatureExtractor(
    model=model,
    params=params,
    layer="stage3",  # Which layer to extract from
)

# Get features
features = extractor.extract(image)
print(f"Feature shape: {features.shape}")

# Multiple layers
extractor = FeatureExtractor(model, params, layer=["stage2", "stage3", "stage4"])
multi_features = extractor.extract(image)
```

## JIT Compilation

Compile for faster inference:

```python
# JIT compile the forward pass
@jax.jit
def predict(params, image):
    return model.apply({'params': params}, image, training=False)

# First call compiles (slow)
logits = predict(params, image)

# Subsequent calls are fast
logits = predict(params, image)
```

## Batched Inference with JIT

```python
# Compile for specific batch size
@jax.jit
def batch_predict(params, images):
    return model.apply({'params': params}, images, training=False)

# Process in batches
batch_size = 32
all_logits = []

for i in range(0, len(images), batch_size):
    batch = images[i:i + batch_size]
    # Pad last batch if needed
    if len(batch) < batch_size:
        pad_size = batch_size - len(batch)
        batch = jnp.concatenate([batch, jnp.zeros((pad_size,) + batch.shape[1:])])
    
    logits = batch_predict(params, batch)
    all_logits.append(logits[:len(images[i:i + batch_size])])

all_logits = jnp.concatenate(all_logits)
```

## Memory Efficient Inference

For large inputs or limited memory:

```python
# Reduce precision
from jax import config
config.update("jax_default_dtype_bits", 16)

# Or use mixed precision manually
params_fp16 = jax.tree_util.tree_map(
    lambda x: x.astype(jnp.float16), params
)

# Process in smaller chunks
def chunked_inference(images, chunk_size=4):
    results = []
    for i in range(0, len(images), chunk_size):
        chunk = images[i:i + chunk_size]
        logits = predict(params, chunk)
        results.append(logits)
    return jnp.concatenate(results)
```

## 3D Data Inference

```python
# For 3D volumes
config = NDSwinConfig.swin_tiny_3d()
model = NDSwinTransformer(config=config)

# Input shape: (batch, channels, depth, height, width)
volume = jnp.ones((1, 1, 32, 64, 64))
logits = model.apply(variables, volume, training=False)
```

## Sliding Window Inference

For inputs larger than training size:

```python
def sliding_window_inference(image, window_size, stride, model, params):
    """Apply model to overlapping windows and average predictions."""
    B, C, H, W = image.shape
    wh, ww = window_size
    sh, sw = stride
    
    # Store predictions
    counts = jnp.zeros((B, num_classes))
    preds = jnp.zeros((B, num_classes))
    
    for y in range(0, H - wh + 1, sh):
        for x in range(0, W - ww + 1, sw):
            window = image[:, :, y:y+wh, x:x+ww]
            logits = model.apply({'params': params}, window, training=False)
            preds = preds + jax.nn.softmax(logits)
            counts = counts + 1
    
    return preds / counts

# Usage
predictions = sliding_window_inference(
    large_image,
    window_size=(224, 224),
    stride=(112, 112),  # 50% overlap
    model=model,
    params=params,
)
```

## Model Export

### To ONNX

```python
from ndswin.inference import export_to_onnx

export_to_onnx(
    model=model,
    params=params,
    path="model.onnx",
    input_shape=(1, 3, 224, 224),
)
```

### To TensorFlow SavedModel

```python
from ndswin.inference import export_to_saved_model

export_to_saved_model(
    model=model,
    params=params,
    path="saved_model/",
    input_shape=(1, 3, 224, 224),
)
```

## Streaming Inference

For real-time applications:

```python
from ndswin.inference import StreamingProcessor

class VideoProcessor:
    def __init__(self, model, params):
        self.predict_fn = jax.jit(
            lambda x: model.apply({'params': params}, x, training=False)
        )
        # Warmup
        dummy = jnp.zeros((1, 3, 224, 224))
        _ = self.predict_fn(dummy)
    
    def process_frame(self, frame):
        # Add batch dimension
        frame = frame[None, ...]
        logits = self.predict_fn(frame)
        return jax.nn.softmax(logits[0])

# Usage
processor = VideoProcessor(model, params)

for frame in video_stream:
    probs = processor.process_frame(frame)
    print(f"Frame prediction: {jnp.argmax(probs)}")
```

## Benchmarking Inference Speed

```python
import time
import jax

def benchmark_inference(model, params, input_shape, num_iterations=100, warmup=10):
    """Benchmark inference speed."""
    dummy_input = jnp.ones(input_shape)
    
    # JIT compile
    predict_fn = jax.jit(
        lambda x: model.apply({'params': params}, x, training=False)
    )
    
    # Warmup
    for _ in range(warmup):
        _ = predict_fn(dummy_input)
    jax.block_until_ready(_)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        output = predict_fn(dummy_input)
    jax.block_until_ready(output)
    elapsed = time.time() - start_time
    
    avg_time = elapsed / num_iterations
    throughput = input_shape[0] / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.1f} samples/sec")
    
    return avg_time, throughput

# Run benchmark
benchmark_inference(model, params, (32, 3, 224, 224))
```

## Tips for Production

1. **Always JIT compile** - Essential for fast inference
2. **Warmup** - First inference is slow due to compilation
3. **Batch wisely** - Larger batches increase throughput
4. **Use XLA** - JAX uses XLA for optimization
5. **Profile** - Use JAX profiler to find bottlenecks
6. **Quantize** - Consider INT8 for deployment
7. **Cache compiled functions** - Avoid recompilation
8. **Use appropriate precision** - FP16 or BF16 for speed
