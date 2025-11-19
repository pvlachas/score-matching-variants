# Experiments

This directory contains the main experiment scripts for different score matching variants.

## Scripts

1. **1_score_matching.py** - Original Score Matching algorithm
   - Learns score function ∇_x log p(x) using full Jacobian
   - Trains ScoreNet on Swiss Roll dataset
   - Generates gradient field visualizations and sampling trajectories

2. **2_sliced_score_matching.py** - Sliced Score Matching
   - Scalable variant using random projections
   - Reduces complexity from O(N²+N) to O(N)
   - Avoids expensive trace computation

3. **3_denoising_score_matching.py** - Denoising Score Matching
   - Learns scores by denoising corrupted data
   - Avoids explicit gradient computation
   - More stable training

4. **4_noise_conditional_score_networks.py** - Noise-Conditional Score Networks (NCSN)
   - Multi-scale score learning with noise conditioning
   - Addresses manifold inconsistency and low-density instability
   - Foundation for modern diffusion models

## Running Experiments

Run individual experiments from the root directory:
```bash
python experiments/1_score_matching.py
python experiments/2_sliced_score_matching.py
python experiments/3_denoising_score_matching.py
python experiments/4_noise_conditional_score_networks.py
```

Or run all experiments:
```bash
python main.py
```

## Results

Results are saved to `../results/{method_name}/` directories.
