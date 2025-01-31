from models.resnet import ResNetMCD, ResNetEnsemble, ResNet_DUQ, resnet18, resnet34, resnet50

import torchvision

import torch.nn as nn
import torch
import utils.constants as keys
import os

import models.robustbench_load as rb_load


'''
    The purpose of this function is to provide a direct interface for loading the already trained
    uncertainty aware models.
'''
def load_model(backbone, uq_technique="None", dataset="cifar10", robustness_level='naive_robust',\
               dropout_rate=None, full_bayesian=False, ensemble_size=5, \
               transform=None,  robust_model=None, device='cpu'):
    # Guard for the supported network architectures
    if backbone not in keys.SUPPORTED_BACKBONES:
        raise Exception(f"{backbone} is not a supported network architecture")

    # Guard for the supported data sets
    if dataset not in keys.SUPPORTED_DATASETS:
        raise Exception(f"{dataset} is not a supported dataset")

    if robustness_level not in keys.ROBUSTNESS_LEVELS:
        raise Exception(f"{robustness_level} is not a supported robustness_level")

    model =None

    # FOR THE PAPER WE USE THIS <----------------------------------------------
    if robustness_level == "semi_robust":
        # Import model from robustbench
        model = rb_load.get_local_model(robust_model, dataset, device)

    elif robustness_level == "naive_robust":
        # Matching the UQ Technique
        if 'dropout' in uq_technique:
            # Obtaining the correct dropout rate
            if dropout_rate is None:
                dropout_rate = keys.DEFAULT_DROPOUT_RATE
            elif dropout_rate not in keys.SUPPORTED_DROPOUT_RATES:
                raise Exception(f"You should select one of the following dropout rates {keys.SUPPORTED_DROPOUT_RATES}")

            # Loading the correct MCD ResNet
            if backbone in keys.SUPPORTED_RESNETS:
                temperature = 1.0  # TODO: Find a more elegant solution

                # Creating the MCD ResNet
                model = ResNetMCD(backbone, pretrained=True,  # default = pretrained TRUE
                                  dropout_rate=dropout_rate,
                                  full_bayesian=full_bayesian,
                                  temperature=temperature,
                                  transform=transform)

                # If the technique is embedded dropout we load the special embedded weights
                if uq_technique == 'embedded_dropout':
                    embedding_type = 'embedded_dropout_full_bayes' if full_bayesian else 'embedded_dropout'
                    dropout_id = int(dropout_rate * 10)
                    embedded_path = os.path.join('models', embedding_type, backbone, f"model_dr{dropout_id}.pt")
                    model.backbone.load_state_dict(torch.load(embedded_path, map_location=torch.device(device)))

            elif backbone in keys.SUPPORTED_VGGS:
                raise Exception("Vgg are not implemented yet!")

            # Preparing the network dropout layers
            set_model_to_eval_activating_dropout(model)

        elif uq_technique == 'deep_ensemble':
            temperature = 1.0
            model = ResNetEnsemble(backbone, ensemble_size, transform=transform)
            model.eval()

        elif uq_technique == 'deterministic_uq':
            model = ResNet_DUQ(transform=transform)
            model.eval()

        elif uq_technique == "None":
            # Import modello normale
            if backbone not in keys.SUPPORTED_RESNETS:
                raise Exception(f"{backbone} is not a supported ResNet.")

            if dataset == "cifar10":
                # Using the pre-trained resnet as backbone architecture, custom load
                if backbone == 'resnet18':
                    model = resnet18(pretrained=True)
                elif backbone == 'resnet34':
                    model = resnet34(pretrained=True)
                elif backbone == 'resnet50':
                    model = resnet50(pretrained=True)

            if dataset == "imagenet":
                # Using the pre-defined resnet as backbone architecture
                if backbone == 'resnet18':
                    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1, progress=False)
                elif backbone == 'resnet34':
                    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1, progress=False)
                elif backbone == 'resnet50':
                    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1, progress=False)
                elif backbone == "Swin-B":                    
                    model = torchvision.models.swin_b(weights=torchvision.models.Swin_B_Weights.IMAGENET1K_V1, progress=False)                
                elif backbone == "ConvNeXt-B":
                    model = torchvision.models.convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1, progress=False)                
                elif backbone == "ConvNeXt-L":
                    model = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1, progress=False)

        else:
            raise Exception(f"{uq_technique} is not a supported uncertainty quantification technique.")

    return model


# Private procedure for set the network to eval while keeping active the dropout layers
def set_model_to_eval_activating_dropout(model):
    # Set to eval mode (no gradients, no batch norm)
    model.eval()

    # Reactivating the dropout layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.training = True
