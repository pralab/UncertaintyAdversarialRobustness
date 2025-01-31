from functools import partial

from torch import nn

from robustbench import load_model

from utils.constants import cifar10_model_dict, imagenet_model_dict

def load_robustbench_model(name: str, dataset: str, threat_model: str, device: str = "cpu", **kwargs) -> nn.Module:
    model = load_model(model_name=name, dataset=dataset, threat_model=threat_model)
    return model


# ---------------- CIFAR-10 ----------------------
_local_cifar_models = {
    'addepalli2022_towards': partial(load_robustbench_model, **cifar10_model_dict["addepalli2022_towards"]),
    'addepalli2022': partial(load_robustbench_model, **cifar10_model_dict["addepalli2022"]),
    'sehwag2021': partial(load_robustbench_model, **cifar10_model_dict["sehwag2021"]),
    'engstrom2019': partial(load_robustbench_model, **cifar10_model_dict["engstrom2019"]),
    'sehwag2021Proxy_ResNest152': partial(load_robustbench_model, **cifar10_model_dict["sehwag2021Proxy_ResNest152"]),
    'pang2022Robustness_WRN70_16': partial(load_robustbench_model, **cifar10_model_dict["pang2022Robustness_WRN70_16"]),
    'gowal2021Improving_28_10': partial(load_robustbench_model, **cifar10_model_dict["gowal2021Improving_28_10"]),
    'kang2021Stable': partial(load_robustbench_model, **cifar10_model_dict["kang2021Stable"]),

    'Peng2023Robust': partial(load_robustbench_model, **cifar10_model_dict["Peng2023Robust"]),
    'Wang2023Better_WRN_70_16': partial(load_robustbench_model, **cifar10_model_dict["Wang2023Better_WRN_70_16"]),
    'Cui2023Decoupled_WRN_28_10': partial(load_robustbench_model, **cifar10_model_dict["Cui2023Decoupled_WRN_28_10"]),
    'Wang2023Better_WRN_28_10': partial(load_robustbench_model, **cifar10_model_dict["Wang2023Better_WRN_28_10"]),
    'Rebuffi2021Fixing_70_16_cutmix_extra': partial(load_robustbench_model,
                                                    **cifar10_model_dict["Rebuffi2021Fixing_70_16_cutmix_extra"]),
    'Gowal2021Improving_70_16_ddpm_100m': partial(load_robustbench_model,
                                                  **cifar10_model_dict["Gowal2021Improving_70_16_ddpm_100m"]),
    'Xu2023Exploring_WRN_28_10': partial(load_robustbench_model, **cifar10_model_dict["Xu2023Exploring_WRN_28_10"]),
    'Sehwag2021Proxy': partial(load_robustbench_model, **cifar10_model_dict["Sehwag2021Proxy"]),

}


def get_local_cifar_model(name: str, dataset: str) -> nn.Module:
    return _local_cifar_models[name]()


# -------------------- IMAGENET -----------------------------
_local_imagenet_models ={
    'salman2020R18': partial(load_robustbench_model, **imagenet_model_dict["salman2020R18"]),
    'wong2020': partial(load_robustbench_model, **imagenet_model_dict["wong2020"]),
    'engstrom2019imgnet': partial(load_robustbench_model, **imagenet_model_dict["engstrom2019imgnet"]),
    'salman2020R50': partial(load_robustbench_model, **imagenet_model_dict["salman2020R50"]),

    'Liu2023convNextL': partial(load_robustbench_model, **imagenet_model_dict["Liu2023convNextL"]),
    'Liu2023swinB': partial(load_robustbench_model, **imagenet_model_dict["Liu2023swinB"]),
    'Liu2023convNextB': partial(load_robustbench_model, **imagenet_model_dict["Liu2023convNextB"]),
    'Liu2023swinL': partial(load_robustbench_model, **imagenet_model_dict["Liu2023swinL"]),
}

def get_local_model(name: str, dataset: str, device: str = "cpu") -> nn.Module:
    print(f"Loading {name}")
    if dataset == 'cifar10':
        return _local_cifar_models[name](device=device)
    elif dataset == 'imagenet':
        return _local_imagenet_models[name](device=device)
