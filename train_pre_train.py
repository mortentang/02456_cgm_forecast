# %%
import datetime
import getpass
import json
import os
from shutil import copyfile

import torch
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

from config import code_path, data_path, figure_path, model_path, result_path
from src.data import DataframeDataLoader
from src.evaluation import evaluateModel
from src.load_data import dataLoader
from src.models.model import DilatedNet
from src.parameter_sets.par_pre_train import *
from src.tools import train_cgm

# %load_ext autoreload
# %autoreload 2


# Paths to data, code, figures, etc. should be set in config.py.
# Initialize the config.py file by copying from config.template.py.
# ---------------------------------------------------------------------
# DEFINE MODEL, PARAMETERS AND DATA
# - Change <par> to the name of file containing your parameters
# - Change <hediaNet> to the name of file containing your model architecture and DilatedNet to the name
#   of your model. Also change in train_cgm and optmizeHypers.py
# ---------------------------------------------------------------------


# Tensorboard log setup
# Create a directory for the model if it doesn't already exist
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
user = getpass.getuser()
model_id = f'id_{current_time}_{user}'
model_path_id = model_path / model_id
model_path_id.mkdir(exist_ok=True, parents=True)
model_figure_path = figure_path / model_id
model_figure_path.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------
# DEFINE DATA
# ---------------------------------------------------------------------
# Define data object
data_pars = {}
data_pars['path'] = data_path
data_pars['train_data'] = train_data
data_pars['test_data'] = test_data
data_pars['validation_data'] = val_data

data_pars['start_date_train'] = start_date_train
data_pars['start_date_test'] = start_date_test
data_pars['start_date_validation'] = start_date_val

data_pars['end_date_train'] = end_date_train
data_pars['end_date_test'] = end_date_test
data_pars['end_date_validation'] = end_date_val


data_obj = dataLoader(data_pars, features, n_steps_past=16,
                      n_steps_future=6,
                      allowed_gap=10,
                      scaler=StandardScaler())


# ---------------------------------------------------------------------
# EXTRACT DATA AND TEST THE MODEL
# ---------------------------------------------------------------------
config = {
    "batch_size": 2,
    "lr": 0.0001,
    "h1": 8,
    "h2": 16,
    "h3": 32,
    "h4": 64,
    "wd": 4e-3,
}


model = DilatedNet(h1=config["h1"],
                   h2=config["h2"],
                   h3=config["h3"],
                   h4=config["h4"],
                   )
model = model.cuda()
# Load training data
trainset, valset = data_obj.load_train_and_val()

train_loader = DataframeDataLoader(
    trainset,
    batch_size=int(config['batch_size']),
    shuffle=True,
    drop_last=True,
)

# Perform a single prediction
data = next(iter(train_loader))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
inputs, targets = data
inputs, targets = inputs.to(device), targets.to(device)
# It is important to permute the dimensions of the input!!
inputs = Variable(inputs.permute(0, 2, 1)).contiguous()

output = model(inputs)

# %%
# ---------------------------------------------------------------------
# TRAING THE MODEL
# ---------------------------------------------------------------------
# Make sure the model archiecture loaded in train_cgm matches the hyper configuration'
pre_train_model = model_path / 'model-folder-name'
train_cgm(config, max_epochs=30, grace_period=5,
          n_epochs_stop=10, data_obj=data_obj, useRayTune=False, checkpoint_dir=pre_train_model)

# Load best model
model_state, optimizer_state = torch.load(code_path / 'src' / 'model_state_tmp' / 'checkpoint')
model.load_state_dict(model_state, strict=False)

# Copy the trained model to model path
copyfile(code_path / 'src' / 'model_state_tmp' / 'checkpoint',
         model_path_id / 'checkpoint')

with open(code_path / 'src' / 'model_state_tmp' / 'hyperPars.json', 'w') as fp:
    json.dump(config, fp)


# %% Evaluate model
# ---------------------------------------------------------------------
# EVALUATE THE MODEL
# ---------------------------------------------------------------------
evaluationConfiguration = {
    'distance': True,
    'hypo': True,
    'clarke': True,
    'lag': True,
    'plotLag': True,
    'plotTimeseries': True
}
# ---------------------------------------------------------------------
evalObject = evaluateModel(data_obj, model)

if evaluationConfiguration['distance']:
    distance = evalObject.get_distanceAnalysis()
if evaluationConfiguration['hypo']:
    hypo = evalObject.get_hypoAnalysis()
if evaluationConfiguration['lag']:
    lag = evalObject.get_lagAnalysis(figure_path=model_figure_path)
if evaluationConfiguration['plotTimeseries']:
    evalObject.get_timeSeriesPlot(figure_path=model_figure_path)
if evaluationConfiguration['clarke']:
    clarkes, clarkes_prob = evalObject.clarkesErrorGrid(
        'mg/dl', figure_path=model_figure_path)

scores = pd.DataFrame(columns=['RMSE', 'MARD', 'MAE',
                               'A', 'B', 'C', 'D', 'E', 'precision', 'recall', 'F1'])
scores.index.name = '[training], test'

scores.loc[str([train_data, test_data])] = [
        distance['rmse'], distance['mard'], distance['mae'],
        clarkes_prob['A'], clarkes_prob['B'], clarkes_prob['C'], clarkes_prob['D'], clarkes_prob['E'],
        hypo['precision'], hypo['recall'], hypo['F1']
    ]

# Save results
result_path.mkdir(exist_ok=True, parents=True)
scores.to_csv(result_path / 'all_scores.csv')
