# New Step2: Residual Refinement with Structure Alignment

## Overview

Stage 2 refines the coarse prediction Δ₀ from Stage 1 by predicting a bounded residual **r**:

```
Δ = Δ₀ + r,    where r = (r_heading, r_range)
```

**Key idea**: instead of generating RGB images (old step2), we use a **structure consistency loss** that compares edge-based representations of current and target views through a differentiable warp, forcing the residual to improve geometric alignment without relying on camera intrinsics (K), depth (D), or any external map.

---

## Network & Training Pipeline (Question E.1)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                │
│                                                                         │
│  I_c ──→ ResNet-18 ──→ feat_c ─┐                                      │
│                                  ├──→ concat(feat_c, feat_t,           │
│  I_t ──→ ResNet-18 ──→ feat_t ─┤      feat_c−feat_t, d) ──→ MLP ──→ r│
│          (shared)               │                                       │
│  Δ₀  ──→ embed MLP ──→ d ─────┘                                       │
│                                                                         │
│  r = (r_θ, r_range)   bounded by tanh × max_correction                │
│  Δ = Δ₀ + r                                                            │
│                                                                         │
│  ╔══════════════════════════════════════════════════════════════════╗   │
│  ║  Loss = λ_pose · L_pose  +  λ_struct(epoch) · L_struct         ║   │
│  ╚══════════════════════════════════════════════════════════════════╝   │
│                                                                         │
│  L_pose:                                                                │
│    smooth_L1( [cos θ_pred, sin θ_pred], [cos θ_gt, sin θ_gt] )        │
│    + smooth_L1( norm_range_pred, norm_range_gt )                       │
│                                                                         │
│  L_struct (multi-scale):                                                │
│    for each scale s ∈ {1.0, 0.5, 0.25}:                               │
│      S_c^s = downsample(SoftEdge(I_c), s)                             │
│      S_t^s = downsample(SoftEdge(I_t), s)                             │
│      S_warped^s = Warp(S_c^s, Δ)     ← grid_sample, differentiable   │
│      DT_t^s = DistanceTransform(binarise(S_t^s))   ← no gradient     │
│      L_dt^s = mean(S_warped^s · DT_t^s)                               │
│    L_struct = mean over scales                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Gradient flow**: `L_struct → S_warped → grid_sample → grid coords → Δ → r → model params`

The DT map is computed from the **target** (no gradient needed); gradients flow through the warp grid coordinates.

---

## A) Differentiable Warp for 2-DoF (range, heading) without K/D

### Option 1: Scale + Translation (`affine_st`) ★ RECOMMENDED

```
tx = tanh(α · heading_rad)        ∈ (-1, 1)
s  = exp(β · range_m / R_max)     > 0

Affine:  [[s,  0,  tx],
          [0,  s,   0]]
```

| Aspect | Detail |
|--------|--------|
| **heading → tx** | Lateral scene shift. `tanh` bounds translation to image width. Smooth gradient everywhere. |
| **range → scale** | `exp` is always positive, symmetric in log-space, smooth. Moving closer = zoom in (s > 1), farther = zoom out (s < 1). |
| **Stability** | Both `tanh` and `exp` have bounded, non-zero derivatives. No singularities. |
| **Default params** | α = 0.5, β = 0.3, R_max = 132.0 |
| **Best for** | Forward-facing drone cameras (heading ≈ horizontal shift). |

### Option 2: Rotation + Scale (`affine_rs`)

```
θ_rot = α · heading_rad
s     = exp(β · range_m / R_max)

Affine:  [[s·cos θ_rot, −s·sin θ_rot, 0],
          [s·sin θ_rot,  s·cos θ_rot, 0]]
```

| Aspect | Detail |
|--------|--------|
| **heading → rotation** | Image-plane rotation. `sin`/`cos` are inherently bounded. |
| **Stability** | Stable, but less physically accurate for oblique forward-facing views. |
| **Best for** | Top-down / nadir drone views where yaw = in-plane rotation. |

### Parameter normalisation

| Parameter | Input range | Normalisation |
|-----------|-------------|---------------|
| heading | [-180°, 180°] → [-π, π] rad | Used directly in radians |
| range | [-132, 132] m | Divided by `NORM_RANGE_MAX` to get [-1, 1] |
| α (heading_warp_scale) | Fixed 0.5 | `tanh(0.5 · π) ≈ 0.92`, so full heading ≈ full image shift |
| β (range_warp_scale) | Fixed 0.3 | `exp(±0.3) ∈ [0.74, 1.35]`, ±26-35% zoom at extremes |

### Why these choices ensure stable gradients

1. **`tanh` bounding**: prevents grid coordinates from going to infinity for large headings.
2. **`exp` for scale**: always positive, smooth, and the log-space symmetry means zoom-in and zoom-out have equal gradient magnitude.
3. **`F.grid_sample` with bilinear + zeros padding**: provides sub-pixel-smooth gradients through the spatial transformer.

---

## B) Structure Representation S(·)

### Recommended: Differentiable Sobel Soft-Edge

```
Pipeline:  RGB → grayscale → Gaussian blur(σ=1.5) → Sobel Gx, Gy
           → magnitude = √(Gx² + Gy²) → per-sample normalise to [0, 1]
```

**Why soft-edge + DT is the best first choice for this task:**

1. **Fully differentiable** — Sobel is a fixed-weight `conv2d`; no special handling needed.
2. **No external dependency** — no pretrained model, no domain-shift risk.
3. **Captures geometry** — building outlines, road edges, horizon lines are preserved; texture is suppressed.
4. **Smooth [0,1] output** — continuous values produce stable gradients through `grid_sample`.
5. **DT provides wide basin of attraction** — even when warped edges are far from target edges, the loss gradient is non-zero (unlike pixel-wise L1 which is zero when edges don't overlap).

### Alternatives considered

| Method | Differentiable? | Pros | Cons |
|--------|----------------|------|------|
| Canny | No (hysteresis) | Sharp edges | Must precompute; no gradient flow |
| Line segments (LSD) | No | Very sparse, geometric | Non-differentiable, sparse |
| Semantic boundary | Needs pretrained model | Semantically meaningful | Heavy model, domain shift |

### Edge extraction hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `edge_sigma` | 1.5 | Gaussian σ before Sobel. Higher = smoother edges, fewer false positives |
| `edge_threshold` | 0.3 | For DT binarisation. Lower = more edges → denser DT but noisier |

---

## C) Structure Consistency Losses

### Distance Transform (DT) Loss — PRIMARY

#### Forward DT Loss

```
E_t    = 1{S(I_t) > τ}                    # binarise target edges
DT_t   = euclidean_distance_transform(1 − E_t)  # scipy EDT, no gradient
DT_t   = DT_t / max(DT_t)                 # normalise to [0, 1]

L_dt_fwd = mean( S_warped · DT_t )
```

**Interpretation**: each warped-edge pixel is penalised proportionally to its distance from the nearest target edge. If edges are well-aligned, `DT_t ≈ 0` at edge locations → low loss.

#### Backward DT Loss (optional, bidirectional)

```
E_w     = 1{S_warped > τ}                 # binarise warped edges (no gradient)
DT_w    = euclidean_distance_transform(1 − E_w)
DT_w    = DT_w / max(DT_w)

L_dt_bwd = mean( S_t · DT_w )
```

**Note**: gradient does NOT flow through `DT_w` (EDT is non-differentiable). This term provides an additional signal: it penalises target edges that are far from any warped edge.

#### Combined

```
L_dt = L_dt_fwd                    # unidirectional (default)
L_dt = 0.5 · (L_dt_fwd + L_dt_bwd)  # bidirectional (--dt_bidirectional)
```

### Chamfer Loss (optional, `--use_chamfer`)

```
P = {coords of top-K pixels where S_warped > τ}     # warped edge points
Q = {coords of top-K pixels where S_t > τ}           # target edge points

L_chamfer = 0.5 · [ mean_p min_q ‖p − q‖₂  +  mean_q min_p ‖p − q‖₂ ]
```

| Property | DT Loss | Chamfer Loss |
|----------|---------|--------------|
| Complexity | O(HW) per sample (EDT) | O(K²) per sample |
| Gradient quality | Smooth, dense | Noisier, sparse |
| Robustness to sparse edges | Good (DT covers all pixels) | Better (direct point matching) |
| Recommended | ★ Primary | Optional supplement |

Default: `--lambda_chamfer 0.05` (small weight if enabled).

---

## D) Training Strategy

### With supervision (Δ_gt available) — `--mode joint` ★ RECOMMENDED

```
L_total = λ_pose · (λ_h · L_heading + λ_r · L_range)
        + λ_struct(epoch) · L_struct_multiscale
```

- `λ_struct(epoch)` ramps linearly from 0 to `λ_struct` over `struct_warmup_epochs`.
- This lets the model first learn a reasonable residual from pose supervision, then the structure loss fine-tunes geometric consistency.
- **Recommended weights**: `λ_pose=1.0, λ_struct=0.1, struct_warmup_epochs=5`.

### Without supervision (no Δ_gt) — `--mode self_supervised`

```
L_total = λ_struct · L_struct_multiscale
```

Mitigations against local optima:

1. **Multi-scale structure** (`--struct_scales 1.0 0.5 0.25`): coarse scales provide gradient even when fine alignment fails.
2. **Robust loss clamping** (`--struct_loss_clamp 5.0`): prevents outlier samples from dominating training.
3. **Curriculum**: optionally start training with only range (freeze heading head) or only heading (freeze range head), then unfreeze both.
4. **Bounded residual**: `tanh` clipping prevents the residual from diverging.

### Avoiding "warp approximation not accurate enough"

| Strategy | How |
|----------|-----|
| Multi-scale structure | Coarse scales tolerate warp inaccuracy |
| Robust loss clamping | Clamp `L_struct` to max value, avoiding gradient explosion |
| λ_struct warmup | Don't trust structure loss early; let pose loss dominate initially |
| Gradient clipping | `--grad_clip 1.0` prevents unstable updates |
| range/heading separate training | Can train heading residual first (warp translation is more accurate), then range |

---

## E) Hyperparameter Recommendations

### Recommended starting configuration

```bash
# Model
--backbone resnet18
--max_heading_residual_deg 45.0    # ±45° max correction
--max_range_residual 40.0          # ±40m max correction

# Structure
--edge_sigma 1.5                   # Gaussian smoothing before Sobel
--edge_threshold 0.3               # DT binarisation threshold

# Warp
--warp_type affine_st              # Scale + Translation
--heading_warp_scale 0.5           # tanh(0.5·π) ≈ 0.92
--range_warp_scale 0.3             # exp(±0.3) ∈ [0.74, 1.35]

# Loss
--mode joint
--lambda_pose 1.0
--lambda_heading 1.0
--lambda_range 1.0
--lambda_struct 0.1
--struct_scales 1.0 0.5 0.25
--struct_warmup_epochs 5
--robust_struct_loss
--struct_loss_clamp 5.0

# Training
--batch_size 64
--lr 1e-4                          # head learning rate
--lr_backbone 1e-5                 # backbone (10× smaller)
--weight_decay 1e-5
--epochs 30
--warmup_epochs 2
--scheduler cosine
--grad_clip 1.0

# Noise simulation (if no step1 train predictions)
--heading_noise_std 40.0           # matches step1's ~40° heading MAE
--range_noise_std 30.0             # matches step1's ~30m range MAE
```

### Tuning guide

| If you see... | Try... |
|---------------|--------|
| Heading improves but range doesn't | Increase `--lambda_range`, decrease `--range_warp_scale` |
| Structure loss doesn't decrease | Increase `--edge_sigma` (smoother edges), use coarser `--struct_scales 1.0 0.5` |
| Training unstable / NaN | Decrease `--lr`, increase `--grad_clip`, decrease `--lambda_struct` |
| Self-supervised mode stuck | Add `--dt_bidirectional`, `--use_chamfer`, try larger `--struct_warmup_epochs` |
| Overfitting | Increase `--weight_decay`, reduce `--max_heading_residual_deg` |

---

## Usage

```bash
# Train (joint mode, recommended)
bash run.sh

# Evaluate
bash run.sh eval
```

## Directory Structure

```
new_step2/
├── config.py           # All hyperparameters
├── dataset.py          # Dataset with step1 prediction loading / noise simulation
├── models/
│   ├── __init__.py
│   ├── structure.py    # Soft-edge extractor (Sobel-based, differentiable)
│   ├── warp.py         # Differentiable 2-DoF warp (affine_st / affine_rs)
│   └── refiner.py      # Residual prediction network (ResNet-18 backbone)
├── losses.py           # DT loss, Chamfer loss, pose losses, combined
├── train.py            # Training loop with curriculum scheduling
├── evaluate.py         # Evaluation with step1 baseline comparison
├── run.sh              # Launch script
└── README.md           # This file
```
