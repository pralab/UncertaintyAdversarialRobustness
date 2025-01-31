
from attacks.standard_attacks import DEVICE, keys
from .standard_attacks import *
import metrics


"""
    This class have the bayesian loss that performs EOT against MC Dropout
    kwargs contains the parameters of the BaseAttack
"""
class BaseBayesianAttack(BaseAttack):
    
    def __init__(self, mc_sample_size_during_attack=20, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mc_sample_size_during_attack = mc_sample_size_during_attack

    def init_loss(self):
        self.loss_fn = loss_functions.BayesianCrossEntropyLoss()
    
    def compute_loss(self):
        if self.mc_sample_size_during_attack != 1:
            self.output = self.model(self.x_adv,
                                     mc_sample_size=self.mc_sample_size_during_attack,
                                     get_mc_output=True)
        else:
            # Eventually, adding the transformation to x
            if self.transform is not None:
                temporary_x_adv = self.transform(self.x_adv)
            else:
                temporary_x_adv = self.x_adv
                
            self.output = self.model(temporary_x_adv).unsqueeze(0)


        # At the first iteration we compute the target (which may require the clean output)
        if (self.x == self.x_adv).all():
            self.clean_output = self.output
            self._set_target()

        loss = self._feed_loss()
        return loss
        

'''
    TODO: Add documentation
'''
class StabilizingAttack(BaseBayesianAttack):
    def __init__(self, mc_sample_size_during_attack=20, **kwargs) -> None:
        super().__init__(mc_sample_size_during_attack, **kwargs)
    
    def init_loss(self):
        self.loss_fn = loss_functions.BayesianCrossEntropyLoss()
    
    def _set_target(self):
        self.target = (metrics.mc_samples_mean(self.clean_output)).argmax(dim=1, keepdim=True).flatten().long()


'''
    TODO: Add documentation
'''
class AutoTargetAttack(BaseBayesianAttack):
    def __init__(self, mc_sample_size_during_attack=20, **kwargs) -> None:
        super().__init__(mc_sample_size_during_attack, **kwargs)
    
    def init_loss(self):
        self.loss_fn = loss_functions.BayesianCrossEntropyLoss()

    # Get the most likely wrong class
    def _set_target(self):
        mean_preds = metrics.mc_samples_mean(self.clean_output)
        self.target = (mean_preds).argmax(dim=1, keepdim=True).flatten()
        mask = (self.target == self.y)
        self.target[mask] = (mean_preds).argsort(dim=1)[:,-2][mask].long()


'''
    TODO: Add documentation
'''
class MinVarAttack(BaseBayesianAttack):
    def __init__(self, mc_sample_size_during_attack=20, **kwargs) -> None:
        super().__init__(mc_sample_size_during_attack, **kwargs)
    
    def init_loss(self):
        self.loss_fn = loss_functions.VarianceLoss(beta=1)

    def _feed_loss(self):
        return self.loss_fn(self.output)


'''
    TODO: Add documentation
'''
class MaxVarAttack(BaseBayesianAttack):
    def __init__(self, mc_sample_size_during_attack=20, **kwargs) -> None:
        super().__init__(mc_sample_size_during_attack, **kwargs)
    
    def init_loss(self):
        self.loss_fn = loss_functions.VarianceLoss(beta=-1)

    def _feed_loss(self):
        return self.loss_fn(self.output)
    


'''
    TODO: Add documentation
'''
class ShakeAttack(BaseBayesianAttack):
    def __init__(self, mc_sample_size_during_attack=20, **kwargs) -> None:
        super().__init__(mc_sample_size_during_attack, **kwargs)
    
    def init_loss(self):
        self.loss_fn = loss_functions.BayesianCrossEntropyLoss(label_smoothing=1)
    
    def _set_target(self):
        self.target = torch.zeros(self.clean_output.shape[1]).to(self.device).long()
        # self.target = (metrics.mc_samples_mean(self.clean_output)).argmax(dim=1, keepdim=True).flatten().long()




# NOTE: --- DO NOT REMOVE THIS CODE BELOW ---
#       It contains the implementation for the Semantic Segmentation attack which still needs to be refactored

#####################################################################


# class PGDBayesianAttack(BaseBayesianAttack):
#     def __init__(self, epsilon=keys.EPSILON, step_size=1, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.optimizer = update_functions.PGDUpdateAndProject(epsilon=epsilon, step_size=step_size)

# '''
#     TODO: Fix this section when refactoring the semantic segmentation part
# '''
# class PGDSemanticSegmentationAttack(PGDBayesianAttack):
#     def __init__(self,
#                  epsilon=keys.EPSILON,
#                  step_size=1,
#                  centered=False,
#                  w=520, h=736,
#                  mask_size=50,
#                  targeted=False,
#                  **kwargs) -> None:
#         super().__init__(epsilon, step_size, **kwargs)
#         self.mask = torch.zeros(size=(1, w, h))
#         self.targeted = targeted
#         # self.mask[:, :mask_size, :mask_size] = 1
#         # self.mask = self.mask.flatten()


#     def compute_loss(self, duq=False):
#         # todo: modularizzare questa parte in modo che se voglio attaccare un non bayesiano lo faccio
#         # output = self.model(self.x_adv, 
#         #                     mc_sample_size=self.mc_sample_size_during_attack,
#         #                     get_mc_output=True)
        
#         output = self.model(x=self.x_adv, 
#                             mc_sample_size=self.mc_sample_size_during_attack,
#                             get_mc_output=True)
        


#         s, b, c, w, h = output.shape    # S, B, 21, W, H  -> 100, 
#         output = torch.transpose(output, 2, 4)
#         output = torch.reshape(output, (s, b*h*w, c))

#         # If I am using deterministic UQ I need to mantain the target unchanged (maybe?)
#         loss = self.loss_fn(output, self.target.flatten().long(), targeted=self.targeted)
            
        
#         # todo: fare qualcosa che raccolga la loss in un file o in un vettore
#         return loss
