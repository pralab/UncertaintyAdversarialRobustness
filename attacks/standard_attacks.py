import torch
from torch.linalg import norm
from time import time

import utils.constants as keys
from . import loss_functions
from . import update_functions
# Queste sono import assolute presumendo che gli script partano dalla root
from utils.utils import get_device
import matplotlib.pyplot as plt
from metrics import var, entropy

DEVICE = get_device()       # Refactor this in using constants

"""
    This class just define:
    - run: the general attack framework
    - compute_loss: with the classic CE, but it is overrided for the MC dropout case
    - update the input using the gradients wrt the computed loss (to be extended)
"""
class BaseAttack:
    
    def __init__(self, model,
                 device=DEVICE,
                 epsilon=keys.BASE_EPSILON,
                 update_strategy='pgd',
                 step_size=None,
                 transform=None) -> None:

        # Setting up the base attack parameters
        self.device = device
        self.model = model
        self.epsilon = epsilon
        self.transform = transform
        self._reset_attack_data()
        self.init_loss()

        # Choosing a suitable optimizer based on the selected update strategy
        if update_strategy == 'pgd':
            self.step_size = step_size
            self.optimizer = update_functions.PGDUpdateAndProject(epsilon=epsilon, step_size=self.step_size)
        elif update_strategy == 'fgsm':
            self.optimizer = update_functions.FGSMUpdateAndProject(epsilon=epsilon)
        else:
            raise Exception(update_strategy, "is not a supported update strategy")
        
        # Defining some parameters for managing the seceval history
        # self.loss_fn = None
        self.loss = None
        
    
    def run(self, x, y=None, iterations=3):

        # Resetting the parameters of the attack
        self._reset_attack_data()

        # Setting up (x,y)
        self.x, self.y = x, y
        self.x_adv = torch.clone(x)

        # Start the evaluation mode and activating the gradients for the adv x
        self.model.eval()
        self.x_adv.requires_grad = True

        # Start the attack iteration
        start = time()
        for _ in range(iterations):
            self.x_adv.requires_grad = True     # TODO: Check if can be removed
            loss = self.compute_loss()          # Compute the loss
            loss.backward()                     # Backpropagate
            self.update_and_project()           # Update x and project onto the epsilon ball
        end = time()
        
        # Saving the attack time
        self.elapsed_time = (end - start)

        # Returning the adversarial example
        self.model.train() # IMPORTANTE
        return self.x_adv.detach()
    
    def compute_loss(self):
        self.output = self.model(self.x_adv)

        # At the first iteration we compute the target (which may require the clean output)
        if (self.x == self.x_adv).all():
            self.clean_output = self.output
            self._set_target()

        # loss = self.loss_fn(self.output, self.target)   # TODO: Find a solution for loss without targets
        loss = self._feed_loss()
        return loss

    def init_loss(self):
        self.loss_fn = loss_functions.MyCrossEntropyLoss()
    
    def update_and_project(self):
        self.x_adv = self.optimizer._update_and_project(self.x_adv, self.x).detach()

    def _reset_attack_data(self):
        self.x, self.x_adv = None, None
        self.output, self.clean_output = None, None
        self.y, self.target = None, None
    
    def _set_target(self):
        self.target = None
    
    def _feed_loss(self):
        return self.loss_fn(self.output, self.target)


'''
    TODO: Add documentation
'''
class DUQAttack(BaseAttack):
    def __init__(self, model, device=DEVICE, epsilon=keys.BASE_EPSILON, update_strategy='pgd', step_size=None) -> None:
        super().__init__(model, device, epsilon, update_strategy, step_size)
        
    def init_loss(self):
        self.loss_fn = loss_functions.RBFLoss()

    def _set_target(self):
        centroids = self.clean_output
        self.target = centroids.argmax(dim=1, keepdim=True).flatten()
        mask = (self.target == self.y)
        self.target[mask] = (centroids).argsort(dim=1)[:,-2][mask]


