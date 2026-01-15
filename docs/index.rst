NDSwin-JAX Documentation
=========================

**N-Dimensional Swin Transformer in JAX**

NDSwin-JAX is a comprehensive implementation of the Swin Transformer
that supports arbitrary dimensions (2D, 3D, 4D, and beyond).

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   configuration

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   training
   inference

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Development

   GitHub Repository <https://github.com/your-org/ndswin-jax>

Quick Example
-------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from ndswin import NDSwinConfig, NDSwinTransformer

   # Create a Swin Transformer for 2D images
   config = NDSwinConfig.swin_tiny_2d()
   model = NDSwinTransformer(config=config)

   # Initialize
   rng = jax.random.PRNGKey(42)
   x = jnp.ones((1, 3, 224, 224))
   variables = model.init(rng, x)

   # Forward pass
   logits = model.apply(variables, x)

Key Features
------------

- **N-Dimensional**: Support for 2D, 3D, 4D, and higher dimensional data
- **JAX/Flax**: Built on JAX for high performance and automatic differentiation
- **Flexible**: Easy configuration for different model sizes and data types
- **Full-featured**: Complete training and inference pipelines
- **Well-Tested**: Comprehensive test suite

Acknowledgments
---------------

- `gerkone/ndswin <https://github.com/gerkone/ndswin>`_: Reference PyTorch implementation.
- Original Swin Transformer: `microsoft/Swin-Transformer <https://github.com/microsoft/Swin-Transformer>`_
- JAX and Flax teams at Google.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
