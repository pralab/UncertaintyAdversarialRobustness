import torch
import matplotlib.pyplot as plt
from utils import *
import torch.nn.functional as F
import numpy as np
import os


#  WARNING: Occhio qua che sono da controllare, ci sono alcune implementazioni che vanno
#           utilizzate e altre che invece non vanno utilizzate

'''
    All the metrics use as input a tensor of 'predictions' with size [S, B, C]
    - S -> The Monte-Carlo Sample Size
    - B -> The Batch Size
    - C -> The number of Classes

    Only the entropy accept a second parameter, which is 'aleatoric_mode'. 
    - True -> Returns the entropy of the mean prediction (which is a measure of aleatoric uncertainty)
    - False -> Returns mean of the Monte-Carlo sample individual entropies (which is a measure of epistemic uncertainty)
'''

# Expected input tensor of shape [S, B]
def get_prediction_with_uncertainty(predictions):
    return mc_samples_mean(predictions), mutual_information(predictions)

# Class-wise shortcut metrics
mc_samples_mean = lambda predictions : torch.mean(predictions, dim=0)
mc_samples_var = lambda predictions : torch.var(predictions, dim=0)
mc_samples_std = lambda predictions : torch.std(predictions, dim=0)

# Point-wise shortcut metrics
var = lambda predictions : torch.mean(mc_samples_var(predictions), dim=1)
std = lambda predictions : torch.mean(mc_samples_std(predictions), dim=1)
mutual_information = lambda predictions : entropy_of_mean_prediction(predictions) - mean_of_sample_entropies(predictions)
entropy = lambda predictions, aleatoric_mode : entropy_of_mean_prediction(predictions) if aleatoric_mode else mean_of_sample_entropies(predictions)

def true_var(predictions):
    # predictions [S, B, C]
    mean_of_square = (predictions * predictions).sum(dim=-1).mean(dim=0)
    mean = mc_samples_mean(predictions)
    square_of_mean = (mean * mean).sum(dim=1)
    var = mean_of_square - square_of_mean
    return var
'''
    Support functions
'''

# [S,B,C] => [S,B]
def __mc_sample_entropies(predictions):
    out = predictions * torch.log(predictions)          # [S,B,C] => [S,B,C]
    # out = torch.where(~torch.isnan(out), out, 0)        # [S,B,C] => [S,B,C]   
    out = torch.where(~torch.isnan(out), out, torch.tensor(0.0, dtype=torch.float32).to(out.device))         
    out = -torch.sum(out, dim=2)                        # [S,B,C] => [S,B]
    return out


# Maps [S,B,C] => [B]
def mean_of_sample_entropies(predictions):
    out = __mc_sample_entropies(predictions)            # [S,B,C] => [S,B]
    out = torch.mean(out, dim=0)                        # [S,B] => [B]
    return out


# Maps [S,B,C] => [B]
def entropy_of_mean_prediction(predictions):
    out = mc_samples_mean(predictions)                  # [S,B,C] => [B,C]
    out = mean_of_sample_entropies(out.unsqueeze(0))    # [B,C] => [1,B,C] => [B]
    return out