# Diarization Simulation

A Python package for simulating speaker diarization with LENA
and [VTC](https://github.com/MarvinLvn/voice-type-classifier) from ground truth vocalization data.

Diarization algorithms segment and classify speech into predefined speaker categories (including child (CHI) other
child (OCH), female adult (FEM), male adult (MAL)).
In Child Development research, these segments are aggregated into vocalization counts (see below) measuring children's
speech output and their speech input in naturalistic daylong recordings.

![](docs/vocalization_counts.png)

However, algorithms make errors (e.g. by confusing speakers for one another) which propagate into the measured
vocalization counts, leading to biases in downstream analyses.
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

## Installation

```bash
# Clone the repository
git clone https://github.com/LAAC-LSCP/diarization-simulation.git
cd diarization-simulation

# Install the package
pip install -e .
```

## Usage

The package provides a command-line tool `diarization-simulate` that simulates detection results:

```bash
diarization-simulate --truth path/to/truth.csv \
                    --output path/to/output.csv \
                    --algo vtc \
                    --samples 1000
```

### Command-line Arguments

| Argument                | Description                                                  | Default   |
|-------------------------|--------------------------------------------------------------|-----------|
| `--truth`               | Path to the ground truth dataset (CSV format)                | Required  |
| `--output`              | Path where the output CSV will be saved                      | Required  |
| `--algo`                | Algorithm to simulate (`vtc` or `lena`)                      | Required  |
| `--samples`             | Number of simulation samples per observation                 | 1000      |
| `--average-hyperpriors` | Use mean values of hyperpriors (mu and alpha)                | True      |
| `--unique-hyperpriors`  | Use fixed hyperpriors throughout all samples                 | True      |
| `--distribution`        | Distribution for vocalization counts (`poisson` or `normal`) | `poisson` |

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
```

### Output Format

The output CSV will contain the following columns:

- `sample`: Sample number (0 to `--samples`-1)
- `observation`: Original observation identifier
- `CHI`: Simulated child vocalization detection
- `OCH`: Simulated other child vocalization detection
- `FEM`: Simulated female adult vocalization detection
- `MAL`: Simulated male adult vocalization detection

## How It Works

The simulation works by:

1. Loading ground truth data about speaker vocalizations
2. Loading pre-computed hyperparameters for the specified algorithm (VTC or LENA)
3. For each sample and observation:
    - Computing speaker detection counts using a statistical model based on the algorithm's hyperparameters
    - The model uses gamma and Poisson distributions to simulate detection probabilities
4. Saving the simulated detection results to a CSV file

## Statistical Model

The simulation uses a hierarchical model where:

- `lambda_ij` ~ Gamma(alpha_ij, mu_ij/alpha_ij) represents detection rates
- Detected vocalizations are generated using one of two distribution options:
    - Poisson distribution: Detected ~ Poisson(lambda_ij * true_ij)
    - Normal distribution: Detected ~ Normal(lambda_ij * true_ij, sqrt(lambda_ij * true_ij / tau))

Where:

- lambda_ij is the detection rate from speaker i to detected speaker j
- true_ij is the true vocalization count for speaker i
- tau is a precision parameter used in the normal distribution option

The Poisson scheme slightly inflates the actual variance, while the normal scheme attempts to capture the correct
variance but may be a poor approximation for small counts.

## Development

### Requirements

You will need Python 3.9+ to run this package.

## Citation

XXX
