# 02456_deeplearning_cgm
A [template](https://github.com/hedia-team/02456_deeplearning_cgmforecast/tree/devBranch) has been given from Hedia, which takes care of running and validating the neural network.

To run the model, the OhioT1DM dataset is needed. This is not publicly available.

# Installation
Using the repo requires installation of `poetry`. Guide to installation can be found [here](https://python-poetry.org/docs/#installation)
This tool manages all the required python packages in a virtual environement. After installation run
`poetry install`from the main directory of the repo.

# Path configuration
Based on `config.template.py` you should create a file `config.py` that defines relevant paths. The `code_path`should refer to this repo, while the other are up to you. It would make sense to locate these outside of the repo to avoid pushing large files to github (otherwiese remember to add these to `.gitignore` before adding.

# Scripts
The repo consist of the following main scripts.
* `model_script.py` Runs the the neural network.
* `testModel.py` Runs a test of the trained model
* `train_pre_train.py` Fine-tune script to further train the neural network
* `optmizeHypers.py` searches for the best hyperparameters given a model and a searching area. You can play around with the searching area searching technqiues to find even better model configurations. 
* `evaluateAll.py` find the best hyperparameters and evalautes the results on a user defined set of data. Create a file with parameters for what data to use in `.src/parameters/evaluateAllPars.py` based on `.src/parameters/evaluateAllPars.template.py`.

The repo also consist of a bunch of helper functions found in `/src/` that load data, evaluate models and so on. 

Finally, one important function is `train_cgm` that id defined in `./src/tools.py`. This function carries out the training of a given model. You are free to change how the training is carried out if you have some good idea. (Which is of course also the case the for any part of the repo).

## IMPORTANT
Remember to load the correct model in the scripts. That is, if you create a new model architecture and save it in `./src/models/myNet.py`, you should change which model is imported in all scripts as well as in `train_cgm`.


# Execution
If you have installed poetry properly, you should be able to run the scripts from the terminal using `poetry run python3 model_script.py`

