import torch
import metrics
import numpy as np
import utils.utils as ut
import utils.utils

'''
    TODO: Anche qua bisogna fare attenzione! Alcune cose sono vecchie implementazioni
'''

METRIC_KEYS = ['ground_truth', 'mean_probs', 'var_probs', 'preds', 'var',
               'mean_of_entropies', 'entropy_of_mean', 'mutual_information', 'true_var']

# entropy_of_mean --> entropy
METRIC_NON_BAYESIAN_KEYS = ['ground_truth', 'mean_probs', 'preds', 'var_probs', 'entropy_of_mean', 'mutual_information']

METRIC_DUQ_KEYS = ['ground_truth', 'preds', 'centroids', 'confidence']

def _eval_duq(model, x, y):
    centroids = model(x)
    ground_truth = y.detach().cpu()
    preds = centroids.argmax(dim=1, keepdim=True).flatten().detach().cpu()
    confidences = torch.max(centroids, dim=1)[0].flatten().detach().cpu()
    partial_results = [ground_truth, preds, centroids, confidences]
    results_dict = {k: v for (k, v) in zip(METRIC_DUQ_KEYS, partial_results)}

    return results_dict


def _eval_bayesian(model, x, y, mc_sample_size=20):
    """
    It returns in output:

    mean_probs (len(x), 10)
    var_probs (len(x), 10)
    preds -> (len(x),)
    var -> (len(x),)
    entropy -> (len(x),)
    mutual_information -> (len(x),)

    """

    output_logits = model(x, mc_sample_size=mc_sample_size, get_mc_output=True)

    output_probs = torch.nn.Softmax(dim=-1)(output_logits)

    ground_truth = y.detach().cpu()
    mean_probs = metrics.mc_samples_mean(output_probs).detach().cpu()
    var_probs = metrics.mc_samples_var(output_probs).detach().cpu()
    preds = mean_probs.argmax(dim=1, keepdim=True).flatten().detach().cpu()
    var = metrics.var(output_probs).detach().cpu()
    mean_of_entropies = metrics.entropy(output_probs, aleatoric_mode=False).detach().cpu()
    entropy_of_mean = metrics.entropy(output_probs, aleatoric_mode=True).detach().cpu()
    mutual_information = metrics.mutual_information(output_probs).detach().cpu()
    true_var = metrics.true_var(output_probs).detach().cpu()

    partial_results = [ground_truth, mean_probs, var_probs, preds, var,
                       mean_of_entropies, entropy_of_mean, mutual_information, true_var]
    results_dict = {k: v for (k, v) in zip(METRIC_KEYS, partial_results)}

    return results_dict


def _eval_non_bayesian(model, x, y, transform=None):
    """
    It returns in output:
    mean_probs (len(x), 10)
    var_probs (len(x), 10)
    preds -> (len(x),)
    mutual_information -> (len(x),)
    """

    if transform:
        x = transform(x)

    output_logits = model(x)
    output_probs = torch.nn.Softmax(dim=-1)(output_logits).detach().cpu()

    ground_truth = y.detach().cpu()
    preds = output_probs.argmax(dim=1).detach().cpu()
    var_probs = torch.var(output_probs, dim=1).cpu()
    entropy_of_mean = metrics.entropy(output_probs.unsqueeze(0), aleatoric_mode=True).detach().cpu()
    # mutual_information = metrics.mutual_information(output_probs.unsqueeze(0)).detach().cpu()
    mutual_information = entropy_of_mean  # NOTE: Correct (?)

    partial_results = [ground_truth, output_probs, preds, var_probs, entropy_of_mean, mutual_information]
    results_dict = {k: v for (k, v) in zip(METRIC_NON_BAYESIAN_KEYS, partial_results)}

    return results_dict


def evaluate_bayesian(model, test_loader, mc_sample_size=20, device='cpu', seed=0):
    model.eval()
    all_results = {k: None for k in METRIC_KEYS}
    utils.utils.set_all_seed(seed)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        results_dict = _eval_bayesian(model, x, y, mc_sample_size)
        for k, v in results_dict.items():
            all_results[k] = torch.cat((all_results[k], v), dim=0) if all_results[k] is not None else v

    return all_results


def evaluate_non_bayesian(model, test_loader, device='cpu', seed=0, transform=None):
    model.eval()
    with torch.no_grad():
        all_results = {k: None for k in METRIC_NON_BAYESIAN_KEYS}
        utils.utils.set_all_seed(seed)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            results_dict = _eval_non_bayesian(model, x, y, transform=transform)
            for k, v in results_dict.items():
                all_results[k] = torch.cat((all_results[k], v), dim=0) if all_results[k] is not None else v

    return all_results


def evaluate_deterministic(model, test_loader, device='cpu', seed=0):
    model.eval()
    all_results = {k: None for k in METRIC_DUQ_KEYS}
    ut.set_all_seed(seed)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        results_dict = _eval_duq(model, x, y)
        for k, v in results_dict.items():
            all_results[k] = torch.cat((all_results[k], v), dim=0) if all_results[k] is not None else v

    return all_results


def evaluate_batch_non_bayesian(model, x, y, outer_dict, transform=None):
    model.eval()
    with torch.no_grad():
        all_results = {k: None for k in METRIC_NON_BAYESIAN_KEYS} if outer_dict is None else outer_dict
        results_dict = _eval_non_bayesian(model, x, y, transform=transform)
        for k, v in results_dict.items():
            all_results[k] = torch.cat((all_results[k], v), dim=0) if all_results[k] is not None else v

    return all_results
