# Tissue Spline Pipeline

A Python pipeline for projecting 2D fluorescence microscopy coordinate data onto a 1D tissue axis, enabling quantitative analysis of spatial distributions across retinal layers.

The pipeline takes spot coordinate data exported from **Imaris** (e.g. DAPI nuclear positions and other fluorescent markers), fits a principal curve through the tissue using a trained **autoencoder neural network**, and classifies each coordinate into its corresponding retinal layer (GCL, INL, ONL) via K-Means clustering on the residuals of projection.

---

## Background

Fluorescence microscopy of retinal sections produces 2D coordinate data for thousands of labelled cells or puncta. Comparing spatial distributions across samples is difficult in raw image coordinates because the retina is curved, tilted, and variable in length between sections. This pipeline solves that problem by:

1. Fitting a smooth 1D axis along the tissue using a neural network principal curve
2. Projecting every coordinate onto that axis to produce a normalised 1D position (`norm1D`, between 0 and 1)
3. Computing the signed perpendicular distance from the axis (`residuals_proj`), which encodes which retinal layer a point belongs to
4. Clustering those residuals into discrete retinal layers (GCL, MidNL, INL, ONL)

This allows direct comparison of cell distributions across samples regardless of tissue orientation or curvature.

---

## File Structure

```
├── tissue_spline_Driver.py     # Top-level script: runs the full pipeline end to end
├── tissue_spline_Utils.py      # All processing functions and the autoencoder model
└── deep_spline/                # Saved pre-trained autoencoder model (required)
```

### `tissue_spline_Driver.py`
The main execution script. Calls utility functions in sequence to:
- Read and clean raw coordinate data
- Train or load the autoencoder
- Fit and trace a spline along the tissue axis
- Cluster coordinates into retinal layers
- Save processed data and diagnostic plots

Edit the config variables at the top of this file before running.

### `tissue_spline_Utils.py`
Contains all processing logic as importable functions. Key stages:

| Function | Purpose |
|---|---|
| `read_imaris_csv` | Reads and renames columns from Imaris CSV export |
| `filter_outliers` | Removes points far from the tissue body using k-NN distances |
| `rescale_data` | Normalises X/Y coordinates to (−1, 1), preserving aspect ratio |
| `reference_channel` | Isolates a single reference marker (typically DAPI) for model training |
| `make_model` | Loads or trains the autoencoder; optionally saves updated weights |
| `autoencoder_model` | Defines the autoencoder architecture used for dimensionality reduction |
| `sort_using_model` | Extracts the bottleneck layer values and sorts coordinates by their 1D position |
| `trace_spline_on_predictions` | Fits a smooth spline through the autoencoder's ordered output |
| `process_df` | Projects all coordinates onto the spline; computes `norm1D` and `residuals_proj` |
| `calc_clusters` | Clusters DAPI residuals into retinal layers; assigns layers to all other objects |

---

## Installation

**Python 3.9+** is recommended.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy tensorflow
```

The pre-trained model (`deep_spline/`) must be present in the working directory for the default `fresh_model=False` mode. If starting without a pre-trained model, set `fresh_model=True` and increase `epochs` (200+ recommended).

---

## Usage

### 1. Configure the Driver Script

Open `tissue_spline_Driver.py` and set the config variables at the top:

```python
save_path = '/path/to/output/directory/'   # where results and plots will be saved
fname = 'my_experiment.csv'                # filename for the processed output CSV
coordinates_path = 'ZEN_ScanRegion_Detailed.csv'  # path to your Imaris export
```

### 2. Prepare Input Data

The pipeline expects a CSV exported from **Imaris** with the following columns (default names, configurable in `read_imaris_csv`):

| CSV Column | Renamed To | Description |
|---|---|---|
| `Position X` | `X` | X coordinate in image space |
| `Position Y` | `Y` | Y coordinate in image space |
| `Unit` | `Unit` | Physical unit of coordinates |
| `Surpass Object` | `Object` | Marker name (e.g. `DAPI`, `EdU`, `BrdU`) |
| `ID` | `ID` | Spot identifier |

The first 3 rows of the CSV are skipped by default (Imaris metadata header). Adjust `skiprows` in `read_imaris_csv` if your export differs.

### 3. Run the Pipeline

```bash
python tissue_spline_Driver.py
```

The pipeline runs in sequence and shows diagnostic plots at each stage. Each plot can be inspected and closed to continue.

### 4. Tune the Outlier Filter

Two filtering steps remove spurious coordinates before model training:

```python
# Step 1: removes points far from the tissue body
df = filter_outliers(data_in, nth_neighbor=30, max_dist=150, show=True)

# Step 2: filters the DAPI reference channel more aggressively
ref_df = reference_channel(df, 'DAPI', nth_neighbor=15, max_dist=0.01, show=True)
```

`nth_neighbor` and `max_dist` control the aggressiveness of filtering. Higher `nth_neighbor` or lower `max_dist` gives stricter filtering. The `show=True` flag displays before/after scatter plots so you can verify the filtering is appropriate for your data.

### 5. Tune the Clustering

```python
df_filt = calc_clusters(temp_df, 
                         n_groups=24, 
                         cluster_seeds=[0.9, 0.0, 1.2, 1.1], 
                         max_proj=0.12, 
                         nth_isolated=100)
```

| Parameter | Description |
|---|---|
| `n_groups` | Number of bins along the 1D axis for local clustering (higher = finer resolution) |
| `cluster_seeds` | Scaling factors for K-Means initialisation; tune these to match your tissue's residual profile |
| `max_proj` | Maximum allowed residual distance; points beyond this are excluded as outliers |
| `nth_isolated` | Removes the `nth_isolated` most spatially isolated points before clustering |

The four clusters correspond to retinal layers in order of their residual distance from the spline: **GCL**, **MidNL (inner nuclear layer boundary)**, **INL**, and **ONL**.

---

## Output

### Processed CSV

Saved to `save_path/Processed_<fname>` with the following columns added to the input data:

| Column | Description |
|---|---|
| `Xnorm`, `Ynorm` | Normalised image coordinates (−1 to 1) |
| `norm1D` | Normalised position along the tissue axis (0 to 1) |
| `Xproj`, `Yproj` | Coordinates of the nearest point on the fitted spline |
| `residuals_proj` | Signed perpendicular distance from the spline (negative = GCL side) |
| `clusters` | Assigned retinal layer (0–3: GCL, MidNL, INL, ONL) |

### Diagnostic Plots

Saved as PDFs to `save_path/plots/`, one set per fluorescent marker:

| File | Contents |
|---|---|
| `<Object>_scatter.pdf` | 2D scatter in image coordinates (top) and 1D projection (bottom), coloured by layer |
| `<Object>_residual.pdf` | Joint plot of `norm1D` vs `residuals_proj` with marginal histograms |
| `<Object>_hist.pdf` | 1D histogram of a specified layer along the tissue axis |

---

## How the Neural Network Works

The autoencoder is a symmetric network with a single-unit bottleneck layer:

```
Input (2D) → Dense → Dense → Dense → [Bottleneck: 1D] → Dense → Dense → Dense → Output (2D)
```

It is trained to reconstruct its own input, which forces the bottleneck to learn a 1D summary of the 2D coordinates. Because the tissue forms an elongated curved shape, the bottleneck learns to represent position along the tissue axis. The bottleneck values are then extracted and used to sort the coordinates, producing an ordered sequence from one end of the tissue to the other.

A smooth spline is then fitted through the ordered predictions to produce a continuous, differentiable tissue axis, which is more accurate than using the raw bottleneck values directly.

---

## Key Parameters at a Glance

| Parameter | Location | Description | Typical Value |
|---|---|---|---|
| `nth_neighbor` | `filter_outliers` | Neighbour rank for outlier detection | 15–30 |
| `max_dist` | `filter_outliers` | Maximum distance to nth neighbour | 100–200 (raw), 0.01 (normalised) |
| `epochs` | `make_model` | Training epochs for the autoencoder | 40 (fine-tuning), 200+ (fresh) |
| `jump_every` | `trace_spline_on_predictions` | Subsampling of predictions before spline fit | 5 |
| `precision` | `trace_spline_on_predictions` | Number of points representing the spline (10^precision) | 4 |
| `n_groups` | `calc_clusters` | Bins along 1D axis for local clustering | 20–30 |
| `max_proj` | `calc_clusters` | Maximum residual distance to retain a point | 0.10–0.15 |

---

## Typical Workflow for a New Sample Type

1. Export spot coordinates from Imaris as CSV
2. Set `save_path`, `fname`, and `coordinates_path` in the driver script
3. Run with `show=True` at each step and inspect the diagnostic plots
4. Adjust `nth_neighbor` and `max_dist` in `filter_outliers` until spurious points are removed
5. Run `make_model` with `fresh_model=False` and `epochs=40` to fine-tune the pre-trained model on your data
6. If the spline does not trace the tissue correctly, increase `epochs` or set `fresh_model=True`
7. Adjust `cluster_seeds` until the four clusters correspond to the expected retinal layers
8. Set `show=False` and `save=True` for a clean final run
