import utils.constants as keys
import os


'''
    -----------------------------------
    --- Documentation for the paths ---
    -----------------------------------

    - experiment_root_path          = "experiments" / experiment_category / robustness_level

    - experiment_model_path
    --- (semantic segmentation)     = dataset / model / uq_technique
    --- (classification iid)        = dataset / model / uq_technique
    --- (classification ood)        = datasetIID-M_datasetOOD-N / model / uq_technique
    
    - experiment_attack_path        = attack_type / attack_name / attack_setup_string

    - experiment_path               = root_path / model_path / attack_path
'''

'''
    TODO: Add documentation
'''
def get_full_experiment_path(experiment_type, robustness_level, dataset, backbone, uq_technique,
                             attack_name, attack_parameters, robust_model=None, dropout_rate=None,
                             ood_dataset=None, iid_size=None, ood_size=None):

    # Composing the three main sub-paths
    root_path = compose_experiment_root_path(experiment_type=experiment_type,
                                             robustness_level=robustness_level)
    model_path = compose_experiment_model_path(experiment_type=experiment_type,
                                               dataset=dataset,
                                               backbone=backbone,
                                               uq_technique=uq_technique,
                                               robust_model=robust_model,
                                               dropout_rate=dropout_rate,
                                               ood_dataset=ood_dataset,
                                               iid_size=iid_size,
                                               ood_size=ood_size)
    attack_path = compose_experiment_attack_path(attack_name=attack_name,
                                                 attack_parameters=attack_parameters)
    
    experiment_path = os.path.join(root_path, model_path, attack_path)
    clean_results_path = os.path.join(root_path, model_path, 'clean_results.pkl')
    adv_results_path = os.path.join(experiment_path, 'adv_results.pkl')
    return  experiment_path, clean_results_path, adv_results_path


'''
    Function used for composing the base experiment root path.
    Such path is formed by joining the following folders:
    - "experiments"         which is the base folder for all the experiments
    - experiment_type       which determines the type of experiment (among classification/segmentation iid/ood)
    - robustness_level      which determines the level of robustness (among 3 levels)
'''
def compose_experiment_root_path(experiment_type, robustness_level):

    # Check for plausibility
    if experiment_type not in keys.EXPERIMENT_CATEGORIES:
        raise Exception(experiment_type, 'is not a valid experiment category.')
    if robustness_level not in keys.ROBUSTNESS_LEVELS:
        raise Exception(robustness_level, 'is not a valid robustness level.')
    
    # Obtaining the path
    path = os.path.join('experiments', experiment_type, robustness_level)
    return path


'''
    Function used for composing base experiment 'model' path.
    Such path incorporates the model and data information.

    In case of IID Clasification and Semantic segmentation it is formed by joining the following folders:
    - dataset               which refers to the dataset used for the experiments
    - backbone              which determines the selected network architecture backbone
    - uq_tecnhique          which indicates the selected uncertainty quantification technique
    And, in case of dropout-based UQ method:
    - dropout_rate          whish is the dropout rate used at test time

    In case of OOD Classification it is formed by joining the following folders:
    - dataset_mixture       which is a string encoding the dataset mixture
    - backbone              which determines the selected network architecture backbone
    - uq_tecnhique          which indicates the selected uncertainty quantification technique
    And, in case of dropout-based UQ method:
    - dropout_rate          whish is the dropout rate used at test time
'''
def compose_experiment_model_path(experiment_type, dataset, backbone, uq_technique, robust_model=None, dropout_rate=None,
                                  ood_dataset=None, iid_size=None, ood_size=None):

    # Check (and eventually correcting) the plausibility of dropout rate
    if dropout_rate is not None and 'dropout' not in uq_technique:
        print("Warning: you are specifying a dropout rate when using a non dropout-based UQ method.")
        print("The dropout rate will be ignored during the folder experiment composition.")
        dropout_rate = None

    # Choosing a different composition strategy based on the experiment type
    if experiment_type == 'classification_ood':
        path = __compose_experiment_model_path_ood(dataset=dataset,
                                                   ood_dataset=ood_dataset,
                                                   iid_size=iid_size,
                                                   ood_size=ood_size,
                                                   backbone=backbone,
                                                   uq_technique=uq_technique,
                                                   dropout_rate=dropout_rate)
    else:
        path = __compose_experiment_model_path_iid(dataset=dataset,
                                                   backbone=backbone,
                                                   uq_technique=uq_technique,
                                                   robust_model=robust_model,
                                                   dropout_rate=dropout_rate)

    return path

    
'''
    Function used for composing base experiment 'model' path in case of iid experiments.
    Such path incorporates the model and data information and is formed by joining the following folders:
    - dataset               which refers to the dataset used for the experiments
    - backbone              which determines the selected network architecture backbone
    - uq_tecnhique          which indicates the selected uncertainty quantification technique
    And, in case of dropout-based UQ method:
    - dropout_rate          whish is the dropout rate used at test time
'''
def __compose_experiment_model_path_iid(dataset, backbone, uq_technique, robust_model=None, dropout_rate=None):

    # Check for plausibility
    if dataset not in keys.SUPPORTED_DATASETS:
        raise Exception(dataset, "is not a supported dataset.")
    if backbone not in keys.SUPPORTED_BACKBONES:
        raise Exception(backbone, "is not a supported backbone.")
    if uq_technique not in keys.SUPPORTE_UQ_METHODS:
        raise Exception(uq_technique, "is not a supported UQ technique.")
    if dropout_rate is not None and dropout_rate not in keys.SUPPORTED_DROPOUT_RATES:
        raise Exception(dropout_rate, "is not a supported dropout rate")
    if robust_model is not None and robust_model not in keys.SUPPORTED_ROBUST_MODEL:
        raise Exception(robust_model, "is not a supported robust model")
    
    # Obtaining the path
    path = os.path.join(dataset, backbone, uq_technique)

    if robust_model is not None:
        path = os.path.join(path, robust_model)

    # If the dropout rate has been specified,
    if dropout_rate is not None:
        path = os.path.join(path, "dropout_rate-" + str(dropout_rate))
    return path


'''
    Function used for composing base experiment 'model' path in case of ood experiments.
    Such path incorporates the model and data information and is formed by joining the following folders:
    - dataset_mixture       which is a string encoding the dataset mixture
    - backbone              which determines the selected network architecture backbone
    - uq_tecnhique          which indicates the selected uncertainty quantification technique
    And, in case of dropout-based UQ method:
    - dropout_rate          whish is the dropout rate used at test time

    where dataset_mixture is a string formed by concatenating the dataset names and sizes separated by an underscore
    (e.g., "cifar10-900_MNIST-600" indicates 900 samples from iid cifar10 plus 600 samples from ood MNIST)
'''
def __compose_experiment_model_path_ood(iid_dataset, ood_dataset, iid_size, ood_size, backbone, uq_technique, dropout_rate=None):
    
    # Check for plausibility
    if iid_dataset not in keys.SUPPORTED_DATASETS:
        raise Exception(iid_dataset, "is not a supported dataset.")
    if ood_dataset not in keys.SUPPORTED_DATASETS:
        raise Exception(ood_dataset, "is not a supported dataset.")
    if backbone not in keys.SUPPORTED_BACKBONES:
        raise Exception(backbone, "is not a supported backbone.")
    if uq_technique not in keys.SUPPORTE_UQ_METHODS:
        raise Exception(uq_technique, "is not a supported UQ technique.")
    if dropout_rate not in keys.SUPPORTED_DROPOUT_RATES:
        raise Exception(dropout_rate, "is not a supported dropout rate")
    
    # Composing the path
    dataset_composition_string = f"{iid_dataset}-{iid_size}_{ood_dataset}-{ood_size}"
    path = os.path.join(dataset_composition_string, backbone, uq_technique)
    return path


'''
    Function used for composing the base experiment "attack" path.
    It encodes the information of the selected attack.
'''
def compose_experiment_attack_path(attack_name, attack_parameters):

    # Unpacking the attack parameters
    (epsilon, norm, strategy, step_size, forward_passes) = attack_parameters

    # Check for plausibility
    if attack_name not in keys.SUPPORTED_ATTACKS:
        raise Exception(attack_name, "is not a supported attack.")
    
    # Obtaining meta-informations from the parameters
    attack_type = __get_attack_type_from_attack_name(attack_name)
    attack_setup_string = __compose_attack_setup_string(epsilon=epsilon,
                                                        norm=norm,
                                                        strategy=strategy,
                                                        step_size=step_size,
                                                        forward_passes=forward_passes)
    
    # Composing the path
    path = os.path.join(attack_type, attack_name, attack_setup_string)
    return path


'''
    TODO: Add documentation
'''
def __get_attack_type_from_attack_name(attack_name):

    atk_type = None
    if attack_name in keys.SUPPORTED_UNDERCONFIDENCE_ATTACKS:
        atk_type = 'U-atk'
    elif attack_name in keys.SUPPORTED_OVERCONFIDENCE_ATTACKS:
        atk_type = 'O-atk'
    else:
        raise Exception(attack_name, "is not a supported attack.")
    
    return atk_type

'''
    TODO: Add documentation
'''
def __compose_attack_setup_string(epsilon, norm, strategy, step_size, forward_passes):

    if strategy not in keys.SUPPORTED_UPDATE_STRATEGIES:
        raise Exception(strategy, "is not a supported attack update strategy.")
    if norm not in keys.SUPPORTED_NORMS:
        raise Exception(norm, "is not a supported norm.")
    setup_string = f"epsilon-{epsilon:.2}___norm-{norm}___strategy-{strategy}___step-{step_size}___forwards-{forward_passes}"
    
    return setup_string

# epsilon-0.034___norm-Linf___update-pgd___step-0.5___forwards-50

# TODO: Occorre capire come gestire l'adversarial training... Lo mettiamo nella stringa di attacco o
# a livelli inferiori della pipeline???