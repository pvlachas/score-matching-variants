# Source Code

This directory contains the core modules used by the experiment scripts.

## Modules

### models.py
- **ScoreNet**: Simple MLP architecture for score estimation
- Architecture: Linear → Softplus → Linear → Softplus → Linear
- Used in standard and sliced score matching experiments

### losses.py
Contains loss functions for different score matching variants:
- **score_matching()**: Full Jacobian-based score matching loss
- **sliced_score_matching()**: Efficient variant using random projections and JVP
- **denoising_score_matching()**: Denoising-based score matching with Gaussian perturbations

### sampling.py
Sampling methods for generating samples from learned score functions:
- **sample_simple()**: Basic gradient ascent sampling
- **sample_langevin()**: Langevin dynamics sampling with noise injection

### data.py
Dataset utilities for the Swiss Roll 2D toy dataset:
- **sample_batch()**: Generate Swiss Roll samples
- **plot_swiss_roll()**: Visualization utility

### utils.py
General utility functions:
- **get_device()**: Automatic device selection (CUDA/MPS/CPU)
- **ensure_fig_dir()**: Create output directories
- **jacobian()**: Efficient Jacobian computation using autograd
- **plot_gradients()**: Visualize learned score fields
- **run_sampling()**: Execute sampling procedures
- **plot_trajectory()**: Visualize sampling trajectories
- **plot_all_trajectories()**: Combined gradient field and trajectory plots

### helper_functions.py
Visualization styling:
- **hdr_plot_style()**: Configure matplotlib with dark background theme
- Sets custom colors, fonts, and figure formatting

## Usage

Import modules from the `src` package:
```python
from src.models import ScoreNet
from src.losses import score_matching
from src.utils import get_device, plot_gradients
from src.sampling import sample_langevin
from src.data import sample_batch
```
