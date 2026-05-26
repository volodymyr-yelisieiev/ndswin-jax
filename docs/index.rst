NDSwin-JAX
==========

NDSwin-JAX is a JAX/Flax implementation of an N-dimensional shifted-window
transformer. The project supports 2D images, 3D voxel inputs, configurable
training runs, sweeps, queues, report artifact extraction, and Hugging Face-ready
weight exports.

Primary Entry Points
--------------------

Use the package CLI directly or the Makefile convenience targets:

* ``ndswin train --config configs/cifar10.json``
* ``ndswin auto-sweep --sweep configs/sweeps/cifar10_hyperparam_sweep.yaml``
* ``make train CONFIG=configs/cifar10.json``
* ``make validate``

Report Artifacts
----------------

The practical-work report and the pinned metric snapshots live under
``report/``. Rebuild the report data and PDF with:

* ``make -C report refresh-data``
* ``make -C report pdf``

For installation, usage examples, and published result links, see the repository
``README.md``.
