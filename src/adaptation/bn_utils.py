import torch.nn as nn


def configure_model_for_tent(model):
    """
    Configures model for TENT adaptation.

    TENT updates only BatchNorm affine parameters:
        - weight / gamma
        - bias / beta

    All other parameters are frozen.
    """

    model.train()

    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)

            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

    return model


def collect_bn_params(model):
    """
    Collects BatchNorm affine parameters for optimization.

    Returns:
        params: list of trainable BatchNorm parameters
        names: names of selected parameters
    """

    params = []
    names = []

    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for param_name, param in module.named_parameters():
                if param_name in ["weight", "bias"]:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")

    return params, names