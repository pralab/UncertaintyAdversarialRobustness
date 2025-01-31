from typing import Union
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import metrics
import pandas as pd
import matplotlib.pyplot as plt


class BaseLoss(nn.Module):
    
    def __init__(self, keep_loss_path=True):
        super(BaseLoss, self).__init__()
        self.keep_loss_path = keep_loss_path
        self.loss_path = {}

    def _add_loss_term(self, key: Union[str, tuple]):
        if isinstance(key, str):
            self.loss_path[key] = []
        else:
            self.loss_path['tot'] = []
            for k in key:
                self.loss_path[k] = []
                
    def _update_loss_path(self, losses, keys):
        for key, loss in list(zip(keys, losses)):
            self.loss_path[key].append(loss.item())
    
    def update_loss_path(self, loss):
        self._update_loss_path((loss,), self.loss_keys)


    def plot_loss(self, ax=None, window=20, fig_path=None):
        # TODO: usare una funzione presa da un qualche visualization.py
        loss_df = pd.DataFrame(self.loss_path)

        if loss_df.shape[0] < window*10:
            window = 1
        
        if isinstance(window, int):
            loss_df = loss_df.rolling(window).mean()

        if ax is None:
            fig, ax = plt.subplots()
        loss_df.plot(ax=ax)
        ax.legend(fontsize=15)
        ax.set_xlabel('iterations')
        
        if fig_path is not None:
            fig.savefig(fig_path)


'''
    TODO: Add documentation
'''
class RBFLoss(BaseLoss):
    def __init__(self, keep_loss_path=True):
        super().__init__(keep_loss_path)
        self.loss_ce_fn = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.mean(torch.tensor(1.0) - input[F.one_hot(target).bool()])
        return loss


'''
    TODO: Add documentation
'''
class VarianceLoss(BaseLoss):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

        # Setting up the visualization variables
        self._add_loss_term('Var')
        self.loss_keys = tuple(self.loss_path.keys())

    def forward(self, input: Tensor) -> Tensor:

        # Computing the variance
        loss = metrics.mc_samples_var(input).mean()
        loss = self.beta * torch.log(loss)              # If beta=1 MinVar attack; if beta=-1 MaxVar attack

        # Updating the loss path for further visualization
        if self.keep_loss_path:
            self.update_loss_path(loss)

        # Returning the loss
        return loss


'''
    TODO: Add documentation
'''
class BayesianCrossEntropyLoss(BaseLoss):
    def __init__(self, targeted=True, label_smoothing=0):
        super().__init__()
        self.beta = 1 if targeted else -1
        self.loss_ce_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Setting up the visualization variables
        self._add_loss_term('CE')
        self.loss_keys = tuple(self.loss_path.keys())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        # Computing the cross-entropy loss
        mean_outs = metrics.mc_samples_mean(input)
        loss = self.beta * self.loss_ce_fn(mean_outs, target.long())    # beta=1 -> get close; beta=-1 -> get far
        
        # Updating the loss path for further visualization
        if self.keep_loss_path:
            self.update_loss_path(loss)

        # Returning the loss
        return loss


'''
    TODO: Add documentation
'''
class BayesianUniformCrossEntropyLoss(BaseLoss):
    def __init__(self, targeted=True):
        super().__init__()
        self.beta = 1 if targeted else -1
        self.loss_ce_fn = nn.CrossEntropyLoss(label_smoothing=1.0)

        # Setting up the visualization variables
        self._add_loss_term('CE')
        self.loss_keys = tuple(self.loss_path.keys())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        # Computing the cross-entropy loss
        mean_outs = metrics.mc_samples_mean(input)
        loss = self.beta * self.loss_ce_fn(mean_outs, target.long())    # beta=1 -> get close; beta=-1 -> get far
        
        # Updating the loss path for further visualization
        if self.keep_loss_path:
            self.update_loss_path(loss)

        # Returning the loss
        return loss


class UncertaintyDivergenceLoss(BaseLoss):
    def __init__(self, alpha=0.5, beta=0.5, keep_loss_path=True):
        super().__init__(keep_loss_path)

        # Setting up learning hyperparameters
        self.alpha = alpha
        self.beta = beta

        # Setting up the visualization variables
        self._add_loss_term(('CE', 'KL-div'))
        self.loss_keys = tuple(self.loss_path.keys())
        self.loss_ce_fn = nn.CrossEntropyLoss()
        self.loss_kl_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
    
    def forward(self, clean_output: Tensor, adv_output: Tensor, target: Tensor) -> Tensor:

        # Computing the loss terms
        cross_entropy_term = self.loss_ce_fn(adv_output, target.long())
        kl_divergence = self.loss_kl_fn(F.log_softmax(clean_output, dim = 1), F.log_softmax(adv_output, dim = 1))
        # loss = (self.alpha * cross_entropy_term + self.beta * kl_divergence)/(self.alpha+self.beta)
        loss = (cross_entropy_term + self.beta * kl_divergence)/(1+ self.beta)
        # Updating the loss path for further visualization
        if self.keep_loss_path:
            self._update_loss_path((loss, cross_entropy_term, kl_divergence), self.loss_keys)

        # Returning the loss
        return loss