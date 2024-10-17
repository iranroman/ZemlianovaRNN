# ZemlianovaRNN    

## Overview
This repository contains the PyTorch implementation of the model described in the 2024 paper ["A Recurrent Neural Network for Rhythmic Timing"](https://www.biorxiv.org/content/10.1101/2024.05.24.595797v1.abstract) by Klavdia Zemlianova, Amit Bose, & John Rinzel. The model is designed to explore and demonstrate the neural mechanisms behind rhythmic timing.

### Status
This project is a **work in progress**. Currently, the repository includes an implementation of a Vanilla RNN (RNN) which is operational. The ZemlianovaRNN model as described in the paper is still under development and will be available in future updates.

## Features
- **Vanilla RNN Implementation:** A standard RNN model which can be used as a baseline or for comparison.
- **Dynamic Configuration:** Model parameters can be dynamically configured using YAML files or overridden via command-line arguments with Fire.
- **Training and Evaluation Scripts:** Scripts to train the model and evaluate its performance are included, with the ability to plot and save results for further analysis.
- **GPU Support:** Efficient training with CUDA, automatically adjusting based on available hardware.

## Getting Started

### Prerequisites
Ensure you have Python 3.6+ and PyTorch installed. Dependencies can be installed via pip:

```bash
pip install -r requirements.txt
```

### Training the Model
To train the Vanilla RNN model, you can use the following command:

```bash
python train.py --model_type RNN
```

### Configuration
Modify the `config.yaml` file to set up different experimental settings or model parameters.

## Contributing
Contributions to the development of ZemlianovaRNN are welcome. Please submit a pull request or open an issue to discuss proposed changes or additions.

## Cite

If you use this code please cite the original paper

```
@article{zemlianova2024recurrent,
  title={A Recurrent Neural Network for Rhythmic Timing},
  author={Zemlianova, Klavdia and Bose, Amitabha and Rinzel, John},
  journal={bioRxiv},
  pages={2024--05},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
