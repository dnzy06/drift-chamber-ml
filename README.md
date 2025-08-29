# Drift Chamber Machine Learning

Comparing classical cluster counting methods for particle identification with resource efficient machine learning models simulated on a chip using hls4ml.

## Overview

This project evaluates the performance of classical algorithms versus machine learning approaches for particle identification in drift chamber data. We use hls4ml to synthesize ML models for FPGA deployment and analyze resource usage, latency, and particle separation performance (kaon vs pion identification).

## Repository Structure

```
├── code/
│   ├── classical_clustercount/     # Traditional cluster counting methods
│   ├── ml_clustercounts/           # Machine learning approaches
│   ├── hls4ml_pipeline/        # FPGA synthesis and resource analysis
│   ├── kaon_pion_separation/    # Performance comparison notebooks
    ├── quantization_hyperparameter_search.py/    # Hyperparameter search
    ├── sparsity_vs_performance.py/    # Testing different pruning degrees
├── data/                        # Dataset files (if included)
├── data_processing/             # Code for data processing
├── models/                      # Trained model files
├── results/                     # Results figures
├── requirements.txt             # Python dependencies
├── .gitattributes               # LFS data tracking
├── hls4ml_setup.sh              # set up hls4ml
└── environment.yml              # Conda environment specification
```

## Installation

### Prerequisites
- Python 3.10+
- Conda/Miniconda

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/dnzy06/drift-chamber-ml.git
cd drift-chamber-ml
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate drift-chamber-ml
```

Or install dependencies manually:
```bash
conda create -n drift-chamber-ml python=3.8
conda activate drift-chamber-ml
pip install hls4ml tensorflow numpy pandas matplotlib jupyter scikit-learn
```

## Usage

### Running the Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate through the notebooks in order:
   - **Classical Clustercount**: Baseline cluster counting methods
   - **ML Clustercount**: Neural network and other ML approaches
   - **HLS4ML Pipeline**: FPGA resource estimation and latency analysis
   - **Kaon-Pion Separation**: Performance comparison and evaluation

### Workflow

1. **Data Processing**: Load and preprocess Garfield++ simulated drift chamber data
2. **Classical Methods**: Implement traditional cluster counting algorithms
3. **ML Training**: Train resource-efficient neural networks
4. **FPGA Synthesis**: Use hls4ml to estimate hardware requirements
5. **Performance Analysis**: Compare accuracy, latency, and resource usage
6. **Particle ID**: Evaluate kaon vs pion separation capabilities

## Dataset
A. Z. Tian et al., "Cluster counting algorithm for the CEPC drift chamber using LSTM and DGCNN," in _Nuclear Science and Techniques_, vol. 36, 2025. doi:10.1007/s41365-025-01670-y

## Results

The project compares:
- **Accuracy**: Classical vs ML particle identification performance
- **Resource Usage**: FPGA LUT, FF, and DSP requirements
- **Latency**: Processing time estimates for real-time applications

## Dependencies

Key libraries used:
- `hls4ml` - High-Level Synthesis for Machine Learning
- `tensorflow` - Neural network training and inference
- `numpy`, `pandas` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - ML utilities and metrics
- `jupyter` - Interactive notebooks

