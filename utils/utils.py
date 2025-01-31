import torch
import pickle
import numpy as np
import random
import os

import torchvision
import utils.constants as keys
from torchvision import transforms
from datetime import datetime
import json

'''
    NOTE: This file still needs to be refactored!
'''


##################################################################################
# TRAINING CHECKPOINTS
##################################################################################

def extract_checkpoints(root_path):
    for f in os.listdir(root_path):
        if 'ckpt' in f:
            model_file_name = f.replace('checkpoint', 'model')
            # model_file_name = f[:-4] + 'pt'
            model_file_name = model_file_name.replace('ckpt', 'pt')
            ckpt = torch.load(os.path.join(root_path, f))
            stripped_ckpt_state_dict = strip_checkpoint(ckpt)
            torch.save(stripped_ckpt_state_dict, os.path.join(root_path, model_file_name))


def strip_checkpoint(checkpoint):
    state_dict = checkpoint['state_dict']
    keys = state_dict.copy().keys()
    for k in keys:
        stripped_k = k[6:]  # Removing the "model." string
        state_dict[stripped_k] = state_dict.pop(k)

    return state_dict


def extract_all_existing_checkpoints(folders=['embedded_dropout', 'deep_ensemble']):
    for method in folders:
        for resn in ['resnet18', 'resnet34', 'resnet50']:
            r = os.path.join('models', method, resn)
            extract_checkpoints(r)


##################################################################################
# DATA UTILS
##################################################################################

# get the preprocessing layers for a given dataset
def get_normalizer(dataset='cifar10'):
    # Obtaining mean and variance for the data normalization
    mean, std = keys.NORMALIZATION_DICT[dataset]
    normalizer = transforms.Normalize(mean=mean, std=std)

    return normalizer


# Function for obtaining a unified division between training, test and validation
def get_dataset_splits(dataset='cifar10', set_normalization=True, ood=False, load_adversarial_set=False, num_advx=None):
    set_all_seed(keys.DATA_SEED)

    # TODO: Find the way for obtaining a ood version
    # Deterioration transformation used for OOD
    deterioration_preprocess = [transforms.ElasticTransform(alpha=250.0), transforms.AugMix(severity=10),
                                transforms.ToTensor()]

    # Defining the validation data-preprocesser
    val_preprocess = [transforms.ToTensor()]

    # Defining the training data-preprocesser with data augmentation
    train_preprocess = [
        # transforms.RandomCrop(32, padding=4), # REMOVED RANDOM CROP
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]

    if set_normalization:
        normalizer = get_normalizer(dataset)
        train_preprocess.append(normalizer)
        val_preprocess.append(normalizer)

    train_preprocess = transforms.Compose(train_preprocess)
    val_preprocess = transforms.Compose(val_preprocess)

    # if corrupted:
    #     val_preprocess = transforms.Compose(deterioration_preprocess)

    if ood:
        dataset = 'cifar100'

    # Loading the original sets
    if dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='datasets', train=True, download=True, transform=train_preprocess)
        test_set = torchvision.datasets.CIFAR10(root='datasets', train=False, download=True, transform=val_preprocess)
    # Loading the original sets
    elif dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root='datasets', train=True, download=True,
                                                  transform=train_preprocess)
        test_set = torchvision.datasets.CIFAR100(root='datasets', train=False, download=True, transform=val_preprocess)
    elif dataset == 'imagenet':

        train_path = './datasets/imagenet/val'

        resized_transform_list = [
            transforms.Resize((224, 224)),  # Modify the size as needed
            transforms.ToTensor()
        ]

        resized_transform = transforms.Compose(resized_transform_list)

        imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=resized_transform)
        train_set, test_set = torch.utils.data.random_split(imagenet_data, [len(imagenet_data)-10000, 10000])

        # train_set = train_preprocess(train_set)
        # test_set = val_preprocess(test_set)

    else:
        raise Exception(f'{dataset} is not supported at the moment. Try using cifar10.')

    # Using 8000 images for test and 2000 for validation
    test_set, validation_set = torch.utils.data.random_split(test_set, [8000, 2000])
    validation_set = test_set

    if load_adversarial_set:
        set_all_seed(keys.DATA_SEED)
        adversarial_test_set, _ = torch.utils.data.random_split(test_set, [num_advx, len(test_set) - num_advx])
        return adversarial_test_set

    return train_set, validation_set, test_set


##################################################################################
# FILE MANAGEMENT 
##################################################################################

get_device = lambda id=0: f"cuda:{id}" if torch.cuda.is_available() else 'cpu'

# TODO: Add documentation
def my_load(path, format='rb'):
    with open(path, format) as f:
        object = pickle.load(f)
    return object


# TODO: Add documentation
def my_save(object, path, format='wb'):
    with open(path, format) as f:
        pickle.dump(object, f)


# Same as os.path.join, but it replaces all '\\' with '/' when running on Windows
def join(*args):
    path = os.path.join(*args).replace('\\', '/')
    return path


# Procedure for setting all the seeds
def set_all_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


##################################################################################
# EXPERIMENTS PATH
##################################################################################
# TODO: to be documented
def get_base_exp_path(root, dataset, uq_technique, backbone):
    exp_path = join(root, dataset, uq_technique, backbone)
    return exp_path


# TODO: to be documented
def get_paths(root, dataset, uq_technique, dropout_rate, backbone,
              eps, atk_type='pgd-pred', step_size=1, num_advx=100,
              pred_w=1, unc_w=1, mc_atk=30):
    # TODO: Why 'base_exp_path' ?
    base_exp_path = get_base_exp_path(root, dataset, uq_technique, backbone)

    # Obtaining the basic experimental setup string
    advx_dir_name = f"eps-{eps:.3f}--step_size-{step_size:.3f}-num_advx-{num_advx}-" \
                    f"pred_w-{pred_w}-unc_w-{unc_w}-mc_atk-{mc_atk}"
    advx_dir_name = f"{advx_dir_name}-dr-{dropout_rate}" if "dropout" in uq_technique else advx_dir_name

    # advx_dir_name = get_advx_dir_name(eps, step_size, num_advx, pred_w, unc_w, mc_atk)
    advx_exp_path = join(base_exp_path, atk_type, advx_dir_name)
    advx_results_path = join(advx_exp_path, 'adv_results.pkl')
    baseline_results_path = join(base_exp_path, 'clean_results.pkl')

    # TODO: return a dictionary instead N string
    # paths = {'exp_path': exp_path,
    #          ''}

    return base_exp_path, advx_exp_path, advx_results_path, baseline_results_path


# TODO: fix the 0 arg error
# def get_advx_dir_name(**kwargs):
#     s = ''
#     for k, v in kwargs.items():
#         if isinstance(v, float):
#             v = f"{v:.3f}"

#         if s == '':
#             s += f"{k}-{v}"
#         else:
#             s += f"-{k}-{v}"


##################################################################################
# OTHERS
##################################################################################

# Takes as inputs a logits vector and perform temperature scaling on that vector: new_l = l/t
def temperature_scaling(logits, temperature):
    return torch.div(logits, temperature)


# Utility function for converting logits to probabilities
def from_logits_to_probs(logits, temperature):
    scaled_logits = temperature_scaling(logits, temperature)
    return torch.nn.functional.softmax(scaled_logits, dim=-1)


# -----------------------------------
def check_kwarg(kwargs):
    import models.robustbench_load as rl

    if kwargs['robustness_level'] == 'naive_robust':
        kwargs["robust_model"] = None

    # THREAT - MODEL CHECK
    if kwargs['robustness_level'] == 'semi_robust':
        kwargs["uq_technique"] = "None"

        if kwargs['norm'] == 'Linf':
            if kwargs['robust_model'] not in keys.LINF_ROBUST_MODELS:
                raise Exception(f"{kwargs['norm']} is not a supported threat model for {kwargs['robust_model']}")

        elif kwargs['norm'] == "L2":
            if kwargs['robust_model'] not in keys.L2_ROBUST_MODELS:
                raise Exception(f"{kwargs['norm']} is not a supported threat model for {kwargs['robust_model']}")

        if kwargs["dataset"] == "cifar10":
            if kwargs["robust_model"] not in keys.CIFAR10_ROBUST_MODELS:
                raise Exception(f"{kwargs['robust_model']} is not a supported *{kwargs['dataset']}* Robust Model.")

        elif kwargs["dataset"] == "imagenet":
            if kwargs["robust_model"] not in keys.IMAGENET_ROBUST_MODELS:
                raise Exception(f"{kwargs['robust_model']} is not a supported *{kwargs['dataset']}* Robust Model.")

        kwargs["backbone"] = rl.cifar10_model_dict[kwargs["robust_model"]]["resnet_type"] if kwargs["dataset"] == "cifar10" else \
            rl.imagenet_model_dict[kwargs["robust_model"]]["resnet_type"]

    if kwargs['uq_technique'] == 'None':
        kwargs["dropout_rate"] = 0.0
        kwargs["mc_samples_eval"] = 1
        kwargs["mc_samples_attack"] = 1
        kwargs["full_bayesian"] = False


    return kwargs