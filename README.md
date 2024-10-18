# ZemlianovaRNN    

## Overview
This repository contains the PyTorch implementation of the model described in the 2024 paper ["A Recurrent Neural Network for Rhythmic Timing"](https://www.biorxiv.org/content/10.1101/2024.05.24.595797v1.abstract) by Klavdia Zemlianova, Amit Bose, & John Rinzel. The model is designed to explore and demonstrate the neural mechanisms behind rhythmic timing.

### Status
This project is actively being developed. The repository now includes fully operational implementations of both the Vanilla RNN and the ZemlianovaRNN models.

## Features
- **Vanilla RNN Implementation:** A standard RNN model which can be used as a baseline or for comparison.
- **ZemlianovaRNN Implementation:** Fully implemented model based on the 2024 paper, designed to explore neural mechanisms of rhythmic timing.
- **Dynamic Configuration:** Model parameters can be dynamically configured using YAML files or overridden via command-line arguments with Fire.
- **Training and Evaluation Scripts:** Scripts to train the models and evaluate their performance are included, with the ability to plot and save results for further analysis.
- **GPU Support:** Efficient training with CUDA, automatically adjusting based on available hardware.

## Getting Started

### Prerequisites
Ensure you have Python 3.6+ and PyTorch installed. Dependencies can be installed via pip:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the ZemlianovaRNN model, use:

```bash
python train.py --model_type ZemlianovaRNN
```

This will save a `best_model.pth` when the training is complete. It will also create a folder called `plots` with examples of model activity for each of the periods included in the training set. 

### Saving Model Outputs for Analysis After Training

After training, you can run inference on the model to save outputs for further analysis using:

```bash
python drive_model_and_save_outputs.py
```

This script processes inputs based on the configured periods, detects peaks in the model’s output, and saves the model outputs, hidden states, and plots. These are stored in a directory named `model_outputs`. Each file is named to indicate which saved model activity corresponds to a specific stimulus period, making it easier to track and analyze the results.

### Configuration
Modify the `config.yaml` file to set up different experimental settings or model parameters.

### Training the Vanilla RNN

To train the Vanilla RNN model, use the following command:

```bash
python train.py --model_type RNN
```


## Contributing
Contributions to the development of ZemlianovaRNN are welcome. Please submit a pull request or open an issue to discuss proposed changes or additions.

## Cite

If you use this code please cite the original paper:

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
