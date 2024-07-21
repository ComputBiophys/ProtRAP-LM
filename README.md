# ProtRAP-LM: Protein Relative Accessibility Prediction through protein Language Model embeddings
## Introduction

We present a novel transformer-based model, ProtRAP-LM, utilizing language model embeddings as input features, to quickly and accurately predict membrane contact probability and relative accessibility for each residue of a given protein sequence.

This package provides an implementation of the membrane contact probability (MCP), relative accessible surface area (RASA), relative lipid accessibility (RLA), relative solvent accessibility (RSA), and relative buried surface area (RBSA) prediction. 

## Usage
### Requirements
1. PyTorch
2. Python package: numpy

`conda create -n ProtRAP-LM python==3.8`

`conda activate ProtRAP-LM`

`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

`pip install numpy==1.24.2`

### Command
#### Prediction
`python ProtRAP-LM.py --input_file ./example/test.fasta --output_dir ./output/ --device cpu`

$input_file is the path of the input file.

$output_dir is the directory of the output file.

$device is the device used in the calculation, 'cpu' for CPU only and 'cuda:0' for GPU support.

## Server
Please try to use our server of ProtRAP-LM predictor at:

http://www.songlab.cn

## References
Wang, L.; Zhang, J.; Wang, D.; Song, C.* Membrane Contact Probability: An Essential and Predictive Character for the Structural and Functional Studies of Membrane Proteins. PLoS Comput. Biol. 2022, 18, e1009972.

Kang, K.; Wang, L.; Song, C*. ProtRAP: Predicting Lipid Accessibility Together with Solvent Accessibility of Proteins in One Run. J. Chem. Inf. Model. 2023, 63, 1058-1065.

The implementation is based on the projects:

[1] https://github.com/facebookresearch/esm#getting-started-with-this-repo-

[2] https://github.com/ComputBiophys/MCP_Predictor

[3] https://github.com/ComputBiophys/ProtRAP
