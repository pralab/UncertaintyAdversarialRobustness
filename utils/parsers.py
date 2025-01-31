# import paths
# from ... import utils
import utils.constants as keys
import argparse

__root_help_message = "The name of the root folder used for containing the experiments." \
                      "By default, the root is set to 'experiments'.\n"

__experiment_type_help_message = "The type of experiment to perform.\n"
__robustness_level_help_message = "The robustness level selected for the experiment.\n"
__dataset_help_message = "The dataset on which the experiments will be conducted.\n"
__backbone_help_message = "The backbone network architecture to use during the experiment.\n"
__uq_technique_help_message = "The selected uncertainty quantification technique.\n"
__dropout_rate_help_message = "In case of dropout-based UQ techniques, this parameter indicates" \
                              "the dropout rate used at inference time.\n"

__attack_loss_help_message = "The loss function used for optimizing the attack.\n"
__attack_update_strategy_help_message = "The attack update strategy.\n"
__norm_help_message = "The selected norm constraining the adversarial perturbation.\n"

__num_attack_iterations_help_message = ""
__mc_samples_attack_help_message = ""
__num_adv_examples_help_message = ""
__batch_size_help_message = ""
__epsilon_help_message = ""

__mc_samples_eval_help_message = ""
__batch_size_eval_help_message = ""

__step_size_help_message = ""
__seed_help_message = ""
__cuda_help_message = ""

__epsilon_min_help_message = ''
__epsilon_max_help_message = ''

__ood_dataset_help_message = ''
__iid_size_help_message = ''
__ood_size_help_message = ''

__robust_model_help_message = ""

'''
    TODO: Implement class 'argument' for manage the default logic of some parameters.
        I) Set dropout rate up to 0.5 when using a dropout-based UQ Technique
        II) Set step size up to a default value when using a pgd update policy
        III) Set iid and ood sizes, along with ood dataset name when performing a 'classification_ood' experiment
        IV) Se si tratta di un ood experiment bisogna impedire l'utilizzo di AutoTargetAttack
        V) Se si usa modello deterministico consentire solo l'attacco "Centroid"
'''
__main_argument_list = [('root', 'experiments', None, __root_help_message),
                        ('experiment_type', 'classification_id', keys.EXPERIMENT_CATEGORIES,
                         __experiment_type_help_message),
                        ('robustness_level', 'semi_robust', keys.ROBUSTNESS_LEVELS, __robustness_level_help_message),
                        ('dataset', 'cifar10', keys.SUPPORTED_DATASETS, __dataset_help_message),
                        ('backbone', 'resnet18', keys.SUPPORTED_BACKBONES, __backbone_help_message),
                        ('uq_technique', 'None', keys.SUPPORTE_UQ_METHODS, __uq_technique_help_message),  # CHANGED
                        ('dropout_rate', 0.3, keys.SUPPORTED_DROPOUT_RATES, __dropout_rate_help_message),
                        ('attack_loss', 'Stab', keys.SUPPORTED_ATTACKS, __attack_loss_help_message),
                        ('attack_update_strategy', 'pgd', keys.SUPPORTED_UPDATE_STRATEGIES,
                         __attack_update_strategy_help_message),
                        ('norm', 'Linf', keys.SUPPORTED_NORMS, __norm_help_message),
                        ('num_attack_iterations', 150, None, __num_attack_iterations_help_message),
                        ('robust_model', 'engstrom2019', (keys.CIFAR10_ROBUST_MODELS+keys.IMAGENET_ROBUST_MODELS), __robust_model_help_message),  # ADDED
                        ('mc_samples_eval', 5, None, __mc_samples_eval_help_message),
                        ('mc_samples_attack', 5, None, __mc_samples_attack_help_message),
                        ('num_adv_examples', 1000, None, __num_adv_examples_help_message),
                        ('batch_size', 256, None, __batch_size_help_message),
                        ('epsilon', keys.BASE_EPSILON * 5, None, __epsilon_help_message),
                        ('step_size', 2e-3, None, __step_size_help_message),

                        ('batch_size_eval', 64, None, __batch_size_eval_help_message),

                        ('seed', 0, None, __seed_help_message),
                        ('re_evaluation_mode', False, None, 'Re_evaluation message'),
                        ('cuda', 1, keys.SUPPORTED_CUDAS, __cuda_help_message)]

# List of arguments used for the seceval
__num_epsilon_steps_help_message = ''
__seceval_argument_list = [('epsilon_min', 1 / 255, None, __epsilon_min_help_message),
                           ('epsilon_max', 8 / 255, None, __epsilon_max_help_message),
                           ('num_epsilon_steps', 8, None, __num_epsilon_steps_help_message)]

# List of arguments used for the OOD setting
__ood_argument_list = [('ood_dataset', 'cifar100', keys.SUPPORTED_DATASETS, __ood_dataset_help_message),
                       ('iid_size', 900, None, __iid_size_help_message),
                       ('ood_size', 600, None, __ood_size_help_message)]

'''
    TODO: Add documentation
'''


def parse_main_classification():
    parser = argparse.ArgumentParser()
    return __add_list_of_arguments_to_parser(parser, __main_argument_list)


'''
    TODO: Add documentation
'''


def add_seceval_parsing(parser):
    return __add_list_of_arguments_to_parser(parser, __seceval_argument_list)


'''
    TODO: Add documentation
'''


def add_ood_parsing(parser):
    return __add_list_of_arguments_to_parser(parser, __ood_argument_list)


'''
    TODO: Add documentation
'''


def __choose_among_list_help_message(parameter_name, supported_values, default_value=None):
    help_message = "This script currently supports the following " \
                   "options for the '{parameter_name}' parameter: {supported_values}.\n" \
                   "If not specified, the default value is set to {default_value}"
    return help_message


'''
    TODO: Add documentation
'''


def __add_list_of_arguments_to_parser(parser, argument_list):
    for argument in argument_list:
        name, default, choices, help = argument
        parser.add_argument(f'-{name}', default=default, type=type(default), choices=choices, help=help)
    return parser


'''
    TODO: Implement this
'''


def __set_up_default_attack_settings():
    pass
