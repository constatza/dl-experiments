# Default Experiment Configuration

This directory contains the default configuration for Feed-Forward Neural Network (FFNN) experiments. The configuration is defined in `ffnn.toml` and controls various aspects of the training pipeline, including model architecture, training loop, data loading, and logging.

## Configuration Sections

### [SESSION]
Global session parameters.
- **seed**: `int`. Random seed for reproducibility across PyTorch, NumPy, and Python random number generators. Default: `1`.
- **precision**: `str`. Floating point precision for calculations (e.g., `"float64"` for double precision). This is managed by PyTorch Lightning's precision settings. Default: `FULL_32` (float32).

### [MODEL]
Defines the neural network architecture.
- **name**: `str`. The class name of the model. Defaults to `NormScaledConstantWidthFFNN`.
  - **Reference**: `dlkit.core.models.nn.ffnn.norm_scaled.NormScaledConstantWidthFFNN`.
  - **Description**: Wraps a `ConstantWidthFFNN` ([Residual Network](https://en.wikipedia.org/wiki/Residual_neural_network) with constant hidden width) with input/output normalization scaling to enforce homogeneous scaling consistency (specifically for $Ax=b$ problems).
- **module_path**: `str`. Python module path where the model class is located (e.g., `"dlkit.nn.ffnn"`).
- **hidden_size**: `int`. Number of neurons in each hidden layer.
- **num_layers**: `int`. Number of hidden layers in the network.
- **Implicit Defaults**:
  - `norm`: Defaults to `"l2"` (Euclidean norm).
  - `activation`: Defaults to `GELU` ([torch.nn.GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)).
  - `dropout`: Defaults to `0.0`.

### [TRAINING]
Configures the training process, optimizer, and scheduler.
- **epochs**: `int`. Number of training epochs. Default: `100`. (Note: This is overridden by `[TRAINING.trainer].max_epochs` if both are present.)
- **monitor_metric**: `str`. Metric to monitor for early stopping. Default: `"val_loss"`.
- **mode**: `str`. Monitoring mode (e.g., `"min"` for metrics like loss, `"max"` for metrics like accuracy). Default: `"min"`.

#### [TRAINING.lr_tuner]
Configuration for PyTorch Lightning's learning rate finder.
- **Reference**: [Lightning Tuner](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html#lightning.pytorch.tuner.tuning.Tuner.lr_find).
- **min_lr**: `float`. Minimum learning rate to test during range search. Default: `1e-8`.
- **max_lr**: `float`. Maximum learning rate to test during range search. Default: `1.0`.
- **num_training**: `int`. Number of learning rate values to test. Default: `30`.
- **mode**: `str`. Search strategy for updating learning rate:
    - `"exponential"`: Increases LR exponentially (recommended). Default.
    - `"linear"`: Increases LR linearly.

#### [TRAINING.trainer]
Arguments passed to the PyTorch Lightning `Trainer`.
- **Reference**: [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html).
- **max_epochs**: `int`. Maximum number of training epochs. Note: This field in `[TRAINING.trainer]` overrides the `epochs` field in `[TRAINING]` if both are present.
- **accelerator**: `str`. Hardware accelerator to use (e.g., `"auto"`, `"gpu"`, `"cpu"`).
- **enable_checkpointing**: `bool`. Whether to enable model checkpointing. Default: `False`.

#### [TRAINING.trainer.callbacks]
List of PyTorch Lightning callbacks.
- **ModelCheckpoint**: Saves model checkpoints based on validation metrics.
  - **Reference**: [ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html).
  - **filename**: `str`. Prefix for checkpoint filenames.
  - **monitor**: `str`. Metric to monitor (e.g., `"val_loss"`).
  - **save_top_k**: `int`. Number of best checkpoints to keep.
  - **save_weights_only**: `bool`. If `true`, only saves model weights (not optimizer state).
  - **every_n_epochs**: `int`. Frequency of checkpoint saving in epochs.
  - **enable_version_counter**: `bool`. Whether to append version number to filenames.

#### [TRAINING.optimizer]
Optimizer configuration.
- **Reference**: [torch.optim](https://pytorch.org/docs/stable/optim.html).
- **name**: `str`. Optimizer class name (e.g., `"AdamW"`).
- **lr**: `float`. Initial learning rate. Default: `1e-3`.

#### [TRAINING.scheduler]
Learning rate scheduler configuration.
- **Reference**: [torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).
- **name**: `str`. Scheduler class name (e.g., `"ReduceLROnPlateau"` - see [torch.optim.lr_scheduler.ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)).
- **factor**: `float`. Factor by which the learning rate will be reduced.
- **patience**: `int`. Number of epochs with no improvement after which learning rate will be reduced. Default: `1000`.
- **min_lr**: `float`. Lower bound on the learning rate. Default: `1e-8`.

#### [TRAINING.metrics]
List of custom metrics used for evaluation.
- **NormalizedVectorNormError**: Computes the mean normalized error $||pred - target|| / ||target||$.
  - **Reference**: `dlkit.core.training.metrics.torchmetrics_wrappers.NormalizedVectorNormError` (wraps [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/)).
  - **norm_ord**: `int`. Order of the norm (e.g., `2` for L2 norm).
  - **vector_dim**: `int`. Dimension along which to compute the norm (default `-1`).

### [DATASET]
Data loading configuration.
- **name**: `str`. Dataset class name. Defaults to `FlexibleDataset`.
  - **Reference**: `dlkit.core.datasets.flexible.FlexibleDataset` (wraps [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)).
  - **Description**: A dataset capable of loading arbitrary sets of feature and target arrays (from files or memory). In this configuration, data is typically injected programmatically from pre-normalized arrays (`normalized.npz`) to avoid double normalization.

### [DATAMODULE]
PyTorch Lightning DataModule configuration.
- **name**: `str`. DataModule class name. Defaults to `InMemoryModule`.
  - **Reference**: `dlkit.core.datamodules.array.InMemoryModule` (wraps [lightning.LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)).
  - **Description**: A DataModule that holds dataset splits (train, val, test) in memory and serves them via DataLoaders.

#### [DATAMODULE.dataloader]
Arguments passed to the PyTorch `DataLoader`.
- **Reference**: [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
- **num_workers**: `int`. Number of subprocesses to use for data loading. Default: `os.cpu_count() - 1` (minimum 0).
- **batch_size**: `int`. Number of samples per batch. Default: `64`.
- **pin_memory**: `bool`. If `true`, the data loader will copy Tensors into device/CUDA pinned memory before returning them. Default: `True`.
- **shuffle**: `bool`. Whether to shuffle data at every epoch (typically `true` for training). Default: `True`.

### [MLFLOW]
MLflow logging configuration.
- **Reference**: [MLflow Documentation](https://mlflow.org/docs/latest/index.html).
- **enabled**: `bool`. Whether to enable MLflow tracking. Default: `False`.
- **server**:
  - **backend_store_uri**: `str`. URI for the MLflow backend store (e.g., SQLite database).
  - **artifacts_destination**: `str`. Directory path for storing artifacts.

### [OPTUNA]
Hyperparameter optimization configuration.
- **Reference**: [Optuna Documentation](https://optuna.readthedocs.io/en/stable/).
- **enabled**: `bool`. Whether to enable Optuna integration. Default: `False`.
