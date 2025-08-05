# Diarization Simulation

<!-- TOC -->

* [Overview](#overview)
* [How It Works](#how-it-works)
* [Statistical Model](#statistical-model)
* [Installation](#installation)
* [Usage](#usage)
* [Synthetic Ground Truth Generation](#synthetic-ground-truth-generation)
* [Typical Workflow](#typical-workflow)
* [Python API](#python-api-1)
* [Citation](#citation)

<!-- TOC -->

A Python package for simulating speaker diarization with LENA
and [VTC](https://github.com/MarvinLvn/voice-type-classifier) from ground truth vocalization data.

Diarization algorithms segment and classify speech into predefined speaker categories (including child (CHI) other
child (OCH), female adult (FEM), male adult (MAL)).
In Child Development and Language Acquisition research, these segments are aggregated into vocalization counts (see
below) measuring children's
speech output and their speech input in naturalistic daylong recordings.

![](docs/vocalization_counts.png)

However, algorithms make errors (e.g. by mistaking speakers for one another) which propagate into the measured
vocalization counts, introducing biases in downstream analyses.
Simulating diarization algorithms can help assess the sensitivity of a statistical analysis to classification errors.
For instance, simulations can help determine whether correlations between speakers' speech quantities are entirely
consistent with spurious correlations due to classification errors.

## Overview

Diarization Simulation is a tool designed to simulate the distortion of vocalization counts from different speakers by
diarization
algorithms. It takes synthetic ground truth data as its input (the true speaker vocalization counts) and simulates
measured vocalization counts
based on the detection and confusion rates of LENA and VTC. The confusion rates of these algorithms were measured on
calibration data consisting of 30 hours of manual annotations.

## How It Works

The simulation works by:

1. Loading synthetic ground truth data (the "true" vocalization counts per speaker and per observation/recording)
2. Loading pre-computed hyperparameters characterizing the behavior of the chosen algorithm (VTC or LENA)
3. For each sample and observation, generating "measured" vocalization counts using a statistical model representing the
   algorithm's behavior.

## Statistical Model

The simulation uses a hierarchical model where:

- Detection/confusion rates $\lambda_{ij}$ follow: $\lambda_{ij} \sim \mathrm{Gamma}(\alpha_{ij}, \mu_{ij}/\alpha_{ij})$
- Detected vocalizations are generated using one of two distribution options:

- The Poisson distribution:

```math
\mathrm{Detected}_{ij} \sim \mathrm{Poisson}(\lambda_{ij} \cdot \mathrm{true}_{i})
```
```math
\mathrm{Detected}_{j} = \sum_i \mathrm{Detected}_{ij}
```

- The Gamma distribution:

```math
\mathrm{Detected}_{ij} \sim \lfloor\mathrm{Gamma}(\alpha, \beta)+0.5\rfloor
```
```math
\mathrm{Detected}_{j} = \sum_i \mathrm{Detected}_{ij}
```

With $\alpha$ and $\beta$ being fixed such that:

```math
\mathbb{E}[\mathrm{Detected}_{ij}] = \lambda_{ij} \cdot \mathrm{true}_{i} \text{ and } \sigma[\mathrm{Detected}_{ij}] = \sqrt{\frac{\lambda_{ij} \cdot \mathrm{true}_{i}}{\tau}}
```

### Parameters

| Parameter               | Description                                             |
|-------------------------|---------------------------------------------------------|
| $\lambda_{ij}$          | Detection rate from speaker $i$ to detected speaker $j$ |
| $\mathrm{true}_{i}$     | True vocalization count for speaker $i$                 |
| $\tau$                  | Underdispersion parameter                               |
| $\alpha_{ij}, \mu_{ij}$ | Shape and scale parameters for the detection rate prior |
| $\alpha, \beta$         | Shape and rate parameters for the gamma detection model |

### Model Characteristics

The original model assumed a Generalized Poisson Distribution, given that the vocalization counts are underdispersed wrt
the Poisson distribution.
However, sampling from this distribution is a bit harder, and the simulation proposes two approximation schemes instead:

- **Poisson scheme**: neglects the underdispersion of the count data
- **Gamma scheme**: better captures the true variance but only approximate for small count data

## Installation

```bash
# Clone the repository
git clone https://github.com/LAAC-LSCP/diarization-simulation.git
cd diarization-simulation

# Install the package
pip install -e .
```

### Requirements

You will need Python 3.8+ to run this package. Key dependencies include:

- pandas
- numpy
- scipy
- numba
- tqdm

For the generation of synthetic ground-truth data, you will also need the following packages:

- cmdstanpy (see installation instructions [here](https://mc-stan.org/cmdstanpy/installation.html))
- ChildProject

## Usage

The package can be used both programmatically (in Python scripts, notebooks, etc.) and via command-line interface.

### Command-line Interface

The main command line interfaced can be accessed through `diarization-simulate`:

```bash
diarization-simulate --truth path/to/truth.csv \
                    --output path/to/output.csv \
                    --algo vtc \
                    --samples 1000 \
                    --distribution poisson
```

#### Command-line Arguments

| Argument                | Description                                                 | Default   |
|-------------------------|-------------------------------------------------------------|-----------|
| `--truth`               | Path to the synthetic truth dataset (in csv format)         | Required  |
| `--output`              | Location of the output file                                 | Required  |
| `--output-format`       | Output file format (`csv`, `parquet`, or `npz`)             | `csv`     |
| `--algo`                | Algorithm to simulate (`vtc` or `lena`)                     | Required  |
| `--samples`             | Number of samples per observation                           | 1000      |
| `--average-hyperpriors` | Use the mean value of the hyperpriors (mu and alpha)        | False     |
| `--unique-hyperpriors`  | Use fixed hyperpriors (mu and alpha) throughout all samples | False     |
| `--distribution`        | Distribution for vocalization counts (`poisson` or `gamma`) | `poisson` |
| `--seed`                | Random seed for reproducibility                             | None      |

### Input Format

The input CSV must contain the following columns:

- `observation`: Unique identifier for each recording/observation
- `CHI`: Child vocalization count
- `OCH`: Other child vocalization count
- `FEM`: Female adult vocalization count
- `MAL`: Male adult vocalization count

Example:

```csv
observation,CHI,OCH,FEM,MAL
1,120,30,200,50
2,90,15,180,70
3,150,25,220,45
```

### Output Format

The output will contain the following columns:

- `sample`: Sample number (0 to `n_samples-1`)
- `observation`: Original observation identifier
- `CHI`: Simulated child vocalization detection
- `OCH`: Simulated other child vocalization detection
- `FEM`: Simulated female adult vocalization detection
- `MAL`: Simulated male adult vocalization detection

Example output:

```csv
sample,observation,CHI,OCH,FEM,MAL
0,1,118,28,195,52
0,2,87,16,175,73
0,3,145,23,215,48
1,1,122,31,198,49
1,2,92,14,182,68
...
```

## Synthetic Ground Truth Generation

The package includes a tool for generating synthetic ground truth data from real corpus annotations using the
`truth-simulate` command-line tool. This is useful when you have sparse manual annotations describing the distribution
of the quantity of speech in multiple recordings and want to generate realistic
ground truth datasets.

### Command-line Interface for Ground Truth Generation

We also provide a tool for generating synthetic datasets reproducing the characteristics of a real corpus.
The corpus must be compatible with the [ChildProject](https://github.com/LAAC-LSCP/ChildProject) python package (which
should also be installed).

```bash
truth-simulate --corpus path/to/corpus \
               --annotator annotation_set_name \
               --output path/to/ground_truth.csv \
               --samples 1000
```

#### Command-line Arguments for `truth-simulate`

| Argument              | Description                                                                                   | Default  |
|-----------------------|-----------------------------------------------------------------------------------------------|----------|
| `--corpus`            | Path to the input ChildProject corpus                                                         | Required |
| `--annotator`         | Annotation set containing the manual annotations                                              | Required |
| `--output`            | Location of the output file                                                                   | Required |
| `--recordings`        | Path to a CSV dataframes containing the list of recordings                                    | None     |
| `--samples`           | Number of samples to generate                                                                 | 1000     |
| `--mode`              | Sample from the mode of the posterior distribution of hyperparameters                         | False    |
| `--show-distribution` | Show the marginal distribution of speech for each speaker according to the manual annotations | False    |

### How Ground Truth Generation Works

The `truth-simulate` tool uses a Bayesian hierarchical model to infer vocalization rate distributions from sparse manual
annotations and then generates complete ground truth datasets. The process works as follows:

1. **Load corpus data**: Reads a ChildProject corpus containing recordings and manual annotations
2. **Extract annotation statistics**: Counts vocalizations per speaker type (CHI, OCH, FEM, MAL) in manually annotated
   segments
3. **Fit hierarchical model**: Uses Stan to fit a Gamma-Poisson model that estimates vocalization rates per speaker
   across the corpus
4. **Generate samples**: Produces synthetic ground truth vocalization counts for all recordings in the corpus

### Ground Truth Output Format

The output CSV contains synthetic ground truth data with the following columns:

- `recording_filename`: Original recording filename
- `observation`: Unique identifier combining recording filename and sample number (e.g., "recording_001.wav,0")
- `CHI`: Simulated child vocalization count
- `OCH`: Simulated other child vocalization count
- `FEM`: Simulated female adult vocalization count
- `MAL`: Simulated male adult vocalization count

Example output:

```csv
recording_filename,observation,CHI,OCH,FEM,MAL
recording_001.wav,"recording_001.wav,0",145,23,198,67
recording_002.wav,"recording_002.wav,0",112,18,176,45
recording_001.wav,"recording_001.wav,1",138,25,203,72
recording_002.wav,"recording_002.wav,1",119,16,181,49
...
```

The output contains KxN rows where K is the number of recordings and N the number of samples requested.

## Typical Workflow

A complete simulation workflow typically involves two steps:

1. **Generate ground truth** from your corpus annotations:

```bash
truth-simulate --corpus /path/to/corpus \
               --annotator human_annotations \
               --output ground_truth.csv \
               --samples 100
```

2. **Simulate diarization** on the generated ground truth:

```bash
diarization-simulate --truth ground_truth.csv \
                    --output simulated_detections.csv \
                    --algo vtc \
                    --samples 100
```

The output `simulated_detections.csv` will contain 100x100xK rows, where K is the number of recordings in the dataset.

### Python API

#### Quick Start

```python
import pandas as pd
from diarization_simulation import simulate_diarization

# Create or load your truth data
truth_df = pd.DataFrame(
    {
        'observation': [1, 2, 3],
        'CHI': [120, 90, 150],
        'OCH': [30, 15, 25],
        'FEM': [200, 180, 220],
        'MAL': [50, 70, 45]
    }
)

# Simulate detections
results = simulate_diarization(
    truth_data=truth_df,
    algorithm="vtc",
    distribution="poisson",
    n_samples=1000,
    random_seed=42
)

print(f"Generated {len(results)} detection samples")
print(results.head())
```

#### Working with DataFrames

```python
# Load your data
truth_data = pd.read_csv("my_truth_data.csv")

# Quick simulation for analysis
results = simulate_diarization(
    truth_data=truth_data,
    algorithm="vtc",
    n_samples=100,
    hyperprior_mode="unique",  # Same hyperpriors for all samples
    verbose=False  # Disable progress bar
)

# Analyze results
mean_detections = results.groupby('observation')[['CHI', 'OCH', 'FEM', 'MAL']].mean()
print("Mean detections per observation:")
print(mean_detections)
```

## Python API

**`simulate_diarization()` function parameters:**

| Parameter         | Type             | Description                                               | Default     |
|-------------------|------------------|-----------------------------------------------------------|-------------|
| `truth_data`      | str or DataFrame | Path to CSV file or pandas DataFrame with truth data      | Required    |
| `algorithm`       | str              | Algorithm to simulate (`"vtc"` or `"lena"`)               | `"vtc"`     |
| `distribution`    | str              | Distribution type (`"poisson"` or `"gamma"`)              | `"poisson"` |
| `n_samples`       | int              | Number of samples to generate per observation             | 1000        |
| `hyperprior_mode` | str              | Hyperprior handling (`"sample"`, `"average"`, `"unique"`) | `"sample"`  |
| `random_seed`     | int or None      | Random seed for reproducibility                           | None        |
| `verbose`         | bool             | Show progress bar                                         | True        |

**Hyperprior modes:**

- `"sample"`: Each sample gets its own hyperpriors (captures algorithm uncertainty)
- `"average"`: Use mean hyperprior values (reduced variance)
- `"unique"`: Same hyperpriors for all samples (minimal variance)

Example workflow:

```python
import pandas as pd
from diarization_simulation import simulate_diarization

# Load your ground truth data
truth_data = pd.read_csv("ground_truth.csv")

# Run simulations with different parameters
algorithms = ["vtc", "lena"]
distributions = ["poisson", "gamma"]
results = {}

for algo in algorithms:
    for dist in distributions:
        key = f"{algo}_{dist}"
        results[key] = simulate_diarization(
            truth_data=truth_data,
            algorithm=algo,
            distribution=dist,
            n_samples=1000,
            random_seed=42  # For reproducibility
        )

# Compare results
for key, result in results.items():
    correlation = result[['CHI', 'FEM']].corr().iloc[0, 1]
    print(f"{key}: CHI-FEM correlation = {correlation:.3f}")
```

## Citation

If you use this package, please mention both of the following references:

```bib
@online{diarization-simulation,
author={Lucas Gautheron},
year=2025,
title={Diarization Simulation: A Python package for simulating speaker diarization with {LENA and VTC} from ground truth vocalization data},
url={https://github.com/LAAC-LSCP/diarization-simulation}
}

@misc{Gautheron2025,
  title = {Classification errors distort findings in automated speech processing: examples and solutions from child-development research},
  url = {http://dx.doi.org/10.31234/osf.io/u925y_v1},
  author = {Gautheron,  Lucas and Kidd,  Evan and Malko,  Anton and Lavechin,  Marvin and Cristia,  Alejandrina},
  year = {2025},
}
```
