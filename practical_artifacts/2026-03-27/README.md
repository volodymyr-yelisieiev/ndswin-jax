# Practical Results Bundle

This directory contains the final 2D and 3D practical-work artifacts copied out of ignored runtime directories on 2026-03-28.

## 2D

- Dataset: `cifar10`
- Final run source: `outputs/cifar10/cifar10_1cd2486d_20260327_215123/`
- Best metric: `val_accuracy = 0.7658834457397461`
- Best epoch: `103`
- Best config copy: `cifar10/best_cifar10_cifar10_0f80c648_20260327_203226_trial013.json`
- Metrics copy: `cifar10/metrics.json`
- Sweep summary copy: `cifar10/summary.json`
- Training log copy: `cifar10/cifar10_1cd2486d_20260327_215123.log`
- Weight copy: `cifar10/final_checkpoint_2d_cifar10_fp16_compressed.npz`

## 3D

- Dataset: local `modelnet40`
- Final run source: `outputs/volume_folder/modelnet40_ec8c1840_20260327_230041/`
- Best metric: `val_accuracy = 0.8420791029930115`
- Best epoch: `42`
- Best config copy: `modelnet40/best_volume_folder_modelnet40_3e7c8e4f_20260327_222840_trial001.json`
- Metrics copy: `modelnet40/metrics.json`
- Sweep summary copy: `modelnet40/summary.json`
- Training log copy: `modelnet40/modelnet40_ec8c1840_20260327_230041.log`
- Weight copy: `modelnet40/final_checkpoint_3d_modelnet40_ckpt_00005000.npz`

## Queue

- Queue results copy: `queue/queue_20260327_182744.json`
- Queue wrapper log copy: `queue/queue_20260327_182743.log`

## Checkpoint note

The copied weight files are retained training checkpoints copied from ignored runtime directories.

For 3D, the copied checkpoint corresponds to the retained final run checkpoint for the successful local `modelnet40` run.

For 2D, the reported best metric came from the trainer restoring the in-memory best validation state at epoch 103 before final evaluation. The copied file `final_checkpoint_2d_cifar10_fp16_compressed.npz` is a compressed fp16 export derived from the retained latest on-disk checkpoint of that run, not a separately saved `best.ckpt`.
