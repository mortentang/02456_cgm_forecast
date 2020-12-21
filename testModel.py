# %%
import datetime
import getpass
import json
import os
from pathlib import Path
from shutil import copyfile

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config import code_path, data_path, figure_path, model_path, result_path
from src.evaluation import evaluateModel
from src.load_data import dataLoader
from src.models.model import DilatedNet
from src.parameter_sets.evaluateAllPars import (GRACE_PERIOD, GRACE_PERIOD_FINAL, MAX_NUM_EPOCHS,
                                                MAX_NUM_EPOCHS_FINAL, N_EPOCHS_STOP,
                                                N_EPOCHS_STOP_FINAL, NUM_SAMPLES, dates, features,
                                                test_data_sequence, train_data_sequence,
                                                val_data_sequence)
from src.tools import train_cgm

scores = pd.DataFrame(columns=['RMSE', 'MARD', 'MAE',
                               'A', 'B', 'C', 'D', 'E', 'precision', 'recall', 'F1'])
scores.index.name = '[training], test'
for i, (train_data, val_data, test_data) in enumerate(zip(train_data_sequence, val_data_sequence, test_data_sequence)):
        start_date_test = dates['start_date_test'][test_data]
        end_date_test = dates['end_date_test'][test_data]

        start_date_train = list(dates['start_date_train'][train_data])
        end_date_train = list(dates['end_date_train'][train_data])
        start_date_val = dates['start_date_val'][val_data]
        end_date_val = dates['end_date_val'][val_data]
        start_date_test = dates['start_date_test'][test_data]
        end_date_test = dates['end_date_test'][test_data]
        # Define data object
        data_pars = {}
        data_pars['path'] = data_path
        data_pars['train_data'] = train_data
        data_pars['test_data'] = test_data
        data_pars['validation_data'] = val_data

        data_pars['start_date_train'] = start_date_train
        data_pars['start_date_test'] = start_date_test
        data_pars['start_date_validation'] = start_date_test

        data_pars['end_date_train'] = end_date_train
        data_pars['end_date_test'] = end_date_test
        data_pars['end_date_validation'] = end_date_test

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
        data_obj = dataLoader(data_pars, features, n_steps_past=16,
                              n_steps_future=6,
                              allowed_gap=10,
                              scaler=StandardScaler())

        test_set = data_obj.load_test_data()
        # Load best model state
        model_state, optimizer_state = torch.load(model_path / 'model-folder-name' / 'checkpoint')
        model.load_state_dict(model_state)

        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        user = getpass.getuser()
        model_id = f'id_{current_time}_{user}'
        model_figure_path = figure_path / model_id
        model_figure_path.mkdir(exist_ok=True, parents=True)

        best_model_path = model_path / model_id
        best_model_path.mkdir(exist_ok=True, parents=True)

        # ---------------------------------------------------------------------
        # EVALUATE THE MODEL
        # ---------------------------------------------------------------------
        evaluationConfiguration = {
            'distance': 1,
            'hypo': 1,
            'clarke': 1,
            'lag': 0,
            'plotLag': 0,
            'plotTimeseries': 0
        }
        # ---------------------------------------------------------------------

        # Define evaluation class
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

        scores.loc[str([test_data])] = [
            distance['rmse'], distance['mard'], distance['mae'],
            clarkes_prob['A'], clarkes_prob['B'], clarkes_prob['C'], clarkes_prob['D'], clarkes_prob['E'],
            hypo['precision'], hypo['recall'], hypo['F1']
        ]

        # Save results
        result_path.mkdir(exist_ok=True, parents=True)
        scores.to_csv(result_path / 'all_scores.csv')
