# Diarization Simulation

A Python package for simulating speaker diarization with LENA and VTC from ground truth vocalization data.

## Overview

Diarization Simulation is a tool designed to simulate how speaker diarization algorithms might detect vocalizations across different speakers. It takes ground truth data about speaker vocalizations and simulates detection results based on statistical models with hyperparameters derived from real-world algorithm performance.
It is trained on more than 30 hours of audio annotated by both humans and LENA/VTC.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/diarization-simulation.git
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

| Argument | Description | Default |
|----------|-------------|--------|
| `--truth` | Path to the ground truth dataset (CSV format) | Required |
| `--output` | Path where the output CSV will be saved | Required |
| `--algo` | Algorithm to simulate (`vtc` or `lena`) | Required |
| `--samples` | Number of simulation samples per observation | 1000 |
| `--average-hyperpriors` | Use mean values of hyperpriors (mu and alpha) | True |
| `--unique-hyperpriors` | Use fixed hyperpriors throughout all samples | True |

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
- Detected vocalizations ~ Poisson(lambda_ij * true_ij) where:
  - lambda_ij is the detection rate from speaker i to detected speaker j
  - true_ij is the true vocalization count for speaker i

## Development

### Requirements

- Python 3.9+
- pandas
- numpy
- scipy
- tqdm
- importlib-resources

### Adding New Algorithms

To add a new algorithm:

1. Create a new NPZ file with the algorithm's hyperparameters in the `diarization_simulation/data/` directory
2. Update the `--algo` choices in `simulate.py` to include your new algorithm

## License

[MIT License](LICENSE)

## Citation

XXX
