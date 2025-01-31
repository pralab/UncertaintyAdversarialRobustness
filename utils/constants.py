# List of supported datasets
SUPPORTED_DATASETS = ['cifar10', 'cifar100', "imagenet"]


# Lists of all the supported Backbones
SUPPORTED_RESNETS = ["resnet18", "resnet34", "resnet50", "resnet152", "resnet_fcn", 'robust_resnet',
                     'ConvNeXt-L', 'ConvNeXt-B', 'Swin-B', 'Swin-L',
                     'RaWideResNet-70-16',
                     'WideResNet-70-16',
                     'WideResNet-28-10', 'WideResNet-34-10']

SUPPORTED_VGGS = []
SUPPORTED_BACKBONES = SUPPORTED_RESNETS + SUPPORTED_VGGS

# Lists of supported UQ methods
SUPPORTE_UQ_METHODS = ["None", 'embedded_dropout', 'injected_dropout', 'deep_ensemble', 'deterministic_uq']

# Lists of supported experiments
ROBUSTNESS_LEVELS = ['naive_robust', 'semi_robust', 'full_robust']
EXPERIMENT_CATEGORIES = ['classification_id', 'classification_ood', 'semantic_segmentation']

# Lists of supported attacks
SUPPORTED_UNDERCONFIDENCE_ATTACKS = ['MaxVar', 'Shake']
SUPPORTED_OVERCONFIDENCE_ATTACKS = ['MinVar', 'AutoTarget', 'Stab', 'Centroid', 'UST']
SUPPORTED_ATTACKS = SUPPORTED_UNDERCONFIDENCE_ATTACKS + SUPPORTED_OVERCONFIDENCE_ATTACKS
SUPPORTED_UPDATE_STRATEGIES = ['pgd', 'fgsm']
SUPPORTED_NORMS = ['Linf', 'L2']

# List of supported dropout rates and ensemble sizes
SUPPORTED_DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
MAXIMUM_ENSEMBLE_SIZE = 5


# The last hyperparameters chosen during the optimization process (investigate on these)
LAST_HYPERPARAMETERS_CHOICES = (2e-3, 150)
BATCH_FOR_EACH_BACKBONE = {
    'resnet18': (150, 400),
    'resnet34': (64, 128),
    'resnet50': (8, 32)
}

LAST_HYPERPARAMETERS_CHOICES_DUQ = (1e-3, 100)
BATCH_FOR_EACH_BACKBONE_DUQ = {
    'resnet18': (1000, 1000),
    'resnet34': (1000, 1000),
    'resnet50': (1000, 1000)
}

# Seed used for data reproducibility. Should always be 42
DATA_SEED = 42
SUPPORTED_CUDAS = [0, 1]

# Attack constants
ROOT = 'experiments_correct'
EPS_BASE = 0.031  # 8/255
BASE_EPSILON = 0.031  # Just a refactoring of the upper constant
OPTIM_ATK_TYPE = ('fgsm', 'pgd')

SEL_LOSS_TERMS = {'pred': (1, 0),
                  'unc': (0, 1),
                  'both': (1, 1)}

# Dictionary containing all the normalization vectors for each data set
NORMALIZATION_DICT = {
    'cifar10': ([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
}

DEFAULT_DROPOUT_RATE = 0.3

# PAPER CITATION ORDER
CIFAR10_ROBUST_MODELS = ['engstrom2019', 'addepalli2022', 'gowal2021Improving_28_10', 'Gowal2021Improving_70_16_ddpm_100m',
                         'Wang2023Better_WRN_28_10', 'Wang2023Better_WRN_70_16', 'sehwag2021', 'sehwag2021Proxy_ResNest152',
                         'Rebuffi2021Fixing_70_16_cutmix_extra', 'kang2021Stable', 'Peng2023Robust', 'addepalli2022_towards',
                         'Cui2023Decoupled_WRN_28_10', 'Xu2023Exploring_WRN_28_10', 'pang2022Robustness_WRN70_16']


CIFAR10_NAIVE_MODELS = ['resnet18', 'resnet34', 'resnet50']


# PAPER CITATION ORDER
IMAGENET_ROBUST_MODELS = ['engstrom2019imgnet', 'salman2020R18', 'salman2020R50', 'wong2020',
                          'Liu2023swinB', 'Liu2023swinL', 'Liu2023convNextB', 'Liu2023convNextL']

IMAGENET_NAIVE_MODELS = ['resnet18', 'resnet50', 'ConvNeXt-L', 'ConvNeXt-B', 'Swin-B']


L2_ROBUST_MODELS = ['sehwag2021', 'engstrom2019', 'augustin2020']
LINF_ROBUST_MODELS = ['addepalli2022', 'addepalli2022_towards', 'sehwag2021', # RESNET-18
                         'engstrom2019', # RESNET-50
                         'sehwag2021Proxy_ResNest152', # RESNET-152
                         'Cui2023Decoupled_WRN_28_10', 'gowal2021Improving_28_10',
                         'Wang2023Better_WRN_28_10', 'Xu2023Exploring_WRN_28_10', # WRN-28-10
                         'Gowal2021Improving_70_16_ddpm_100m', 'kang2021Stable',
                         'pang2022Robustness_WRN70_16', 'Rebuffi2021Fixing_70_16_cutmix_extra',
                         'Wang2023Better_WRN_70_16', # WRN-70-10
                         'Peng2023Robust',  # RaWRN-70-16

                      'salman2020R18', # RESNET-18
                          'engstrom2019imgnet', 'salman2020R50', 'wong2020', # RESNET-50
                          'Liu2023convNextL', 'Liu2023swinB', # SWIN
                          'Liu2023convNextB', 'Liu2023swinL' # CONVNEXT
                      ]

SUPPORTED_ROBUST_MODEL = CIFAR10_ROBUST_MODELS + IMAGENET_ROBUST_MODELS


cifar10_model_dict = dict(
    addepalli2022={
    'name': 'Addepalli2022Efficient_RN18',  # ResNet-18
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf',
    'resnet_type': 'resnet18'
    },
    addepalli2022_towards={
    'name': 'Addepalli2021Towards_RN18',  # ResNet-18
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf',
    'resnet_type': 'resnet18'
    },
    sehwag2021={
    'name': 'Sehwag2021Proxy_R18',  # ResNet-18
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf',  # Available [Linf, L2]
    'resnet_type': 'resnet18'
    },
    engstrom2019={
    'name': 'Engstrom2019Robustness',  # RESNET50
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf',  # training threat model. Available [Linf, L2]
    'resnet_type': 'resnet50'
    },
    sehwag2021Proxy_ResNest152={
        'name': 'Sehwag2021Proxy_ResNest152',  # RESNET50
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',  # training threat model
        'resnet_type': 'resnet152'
        },
    pang2022Robustness_WRN70_16={
        'name': 'Pang2022Robustness_WRN70_16',  # RESNET50
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',  # training threat model
        'resnet_type': 'WideResNet-70-16'
    },
    gowal2021Improving_28_10={
        'name': 'Gowal2021Improving_28_10_ddpm_100m',  # RESNET50
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',  # training threat model
        'resnet_type': 'WideResNet-28-10'
    },
    kang2021Stable={
        'name': 'Kang2021Stable',  # RESNET50
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',  # training threat model
        'resnet_type': 'WideResNet-70-16'
    },
    ######## New Entries #############
    Peng2023Robust={
        'name': 'Peng2023Robust',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'RaWideResNet-70-16'
    },
    Wang2023Better_WRN_70_16={
        'name': 'Wang2023Better_WRN-70-16',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-70-16'
    },
    Cui2023Decoupled_WRN_28_10={
        'name': 'Cui2023Decoupled_WRN-28-10',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-28-10'
    },
    Wang2023Better_WRN_28_10={
        'name': 'Wang2023Better_WRN-28-10',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-28-10'
    },
    Rebuffi2021Fixing_70_16_cutmix_extra={
        'name': 'Rebuffi2021Fixing_70_16_cutmix_extra',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-70-16'
    },
    Gowal2021Improving_70_16_ddpm_100m={
        'name': 'Gowal2021Improving_70_16_ddpm_100m',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-70-16'
    },
    Xu2023Exploring_WRN_28_10={
        'name': 'Xu2023Exploring_WRN-28-10',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-28-10'
    },
    Sehwag2021Proxy={
        'name': 'Sehwag2021Proxy',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-34-10'
    },
)

imagenet_model_dict = dict(
    salman2020R18={
    'name': 'Salman2020Do_R18',  # ResNet-18
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf',
    'resnet_type': 'resnet18'
    },
    wong2020={
    'name': 'Wong2020Fast',  # ResNet-50
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf',
    'resnet_type': 'resnet50'
    },
    engstrom2019imgnet={
    'name': 'Engstrom2019Robustness',  # ResNet-50
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf',
    'resnet_type': 'resnet50'
    },
    salman2020R50={
    'name': 'Salman2020Do_R50',  # ResNet-50
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf',
    'resnet_type': 'resnet50'
    },
    Liu2023convNextL={
        'name': 'Liu2023Comprehensive_ConvNeXt-L',
        'source': 'robustbench',
        'dataset': 'imagenet',
        'threat_model': 'Linf',
        'resnet_type': 'ConvNeXt-L'
        },
    Liu2023swinB={
            'name': 'Liu2023Comprehensive_Swin-B',
            'source': 'robustbench',
            'dataset': 'imagenet',
            'threat_model': 'Linf',
            'resnet_type': 'Swin-B'
        },
    Liu2023convNextB={
        'name': 'Liu2023Comprehensive_ConvNeXt-B',
        'source': 'robustbench',
        'dataset': 'imagenet',
        'threat_model': 'Linf',
        'resnet_type': 'ConvNeXt-B'
    },
    Liu2023swinL={
        'name': 'Liu2023Comprehensive_Swin-L',
        'source': 'robustbench',
        'dataset': 'imagenet',
        'threat_model': 'Linf',
        'resnet_type': 'Swin-L'
    },
)