# model 

The `model` directory contains trained model checkpoints and code for training and testing the model.

## Model checkpoints
The last model checkpoint for encoder, decoder, and supertag encoder are included from each model run. The directory for each run generally indicates the model parameters used for that run.

## Training and testing scripts
- `model.py` This contains the actual PyTorch model including encoder, decoder, and supertag encoder, as well as functions for training and evaluation. *To train the model, simply change the parameter values at the top of the file to reflect the desired parameters and training data, then run* `python model.py`. 
- `test.py` contains code to test a trained paraphrase model. *To test a model, change the parameter values at the top to match those used during training and the path to test data, then run* `python test.py`.
- `attention.py` This file contains implementations of various attention mechanisms provided by Prof. Robert Frank.
- `model_results.md` This file contains a description of model performances with various input orderings and model sizes.
