import torch
import torch.nn as nn
import os
import utils.constants as keys
from utils.utils import temperature_scaling
import torchvision
from models.fcn import FCN_ResNet50_Weights, fcn_mcd_resnet50

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
            dropout_rate=0.0
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)  # Added Dropout

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.dropout(out)  # Added Dropout

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
            dropout_rate=0.0
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)  # Added Dropout

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)  # Added Dropout

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.dropout(out)  # Added Dropout

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            num_classes=10,
            dropout_rate=0.0,  # Dropout rate to inject / embed on the network
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            full_bayesian=False,  # Determines if use dropout in each layer
            semantic_segmentation=False  # Determines if the task is semantic segmentation or classification
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # REFACTORING-FALL: Questa è la sezione incriminata!Questa variabile per ora statica
        # permette di shiftare da semantic segmentation a classificazione;
        # TODO: occorre modularizzare questo passaggio, propagando le modifiche
        # anche alle altre parti di codice
        # --- BEGIN ---
        semantic_segmentation = False
        if semantic_segmentation:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
        # --- END ---

        # Adding the dropout
        self.full_bayesian = full_bayesian
        self.full_dropout = dropout_rate if full_bayesian else 0.0
        self.dropout_rate = 0.0 if full_bayesian else dropout_rate
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.full_dropout = dropout_rate if full_bayesian else 0.0
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=self.full_dropout)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       dropout_rate=self.full_dropout)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       dropout_rate=self.full_dropout)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       dropout_rate=self.full_dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, dropout_rate=0.0):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                dropout_rate=dropout_rate
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if not self.full_bayesian:
            x = self.dropout_layer(x)

        x = self.layer2(x)
        if not self.full_bayesian:
            x = self.dropout_layer(x)

        x = self.layer3(x)
        if not self.full_bayesian:
            x = self.dropout_layer(x)

        x = self.layer4(x)
        if not self.full_bayesian:
            x = self.dropout_layer(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _ovewrite_named_param(kwargs, param: str, new_value) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def _resnet(arch, block, layers, weights, pretrained, progress, device, dropout_rate=0.0, full_bayesian=False,
            **kwargs):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, dropout_rate=dropout_rate, full_bayesian=full_bayesian, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/pretrained_baselines/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def resnet18(weights=None, pretrained=False, progress=True, device="cpu", dropout_rate=0.0, full_bayesian=False,
             **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], None, pretrained, progress, device, **kwargs, dropout_rate=dropout_rate,
        full_bayesian=full_bayesian
    )


def resnet34(weights=None, pretrained=False, progress=True, device="cpu", dropout_rate=0.0, full_bayesian=False,
             **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet34", BasicBlock, [3, 4, 6, 3], None, pretrained, progress, device, **kwargs, dropout_rate=dropout_rate,
        full_bayesian=full_bayesian
    )


def resnet50(weights=None, pretrained=False, progress=True, device="cpu", dropout_rate=0.0, full_bayesian=False,
             **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], weights, pretrained, progress, device, **kwargs,
        dropout_rate=dropout_rate, full_bayesian=full_bayesian
    )


# def mcd_resnet(resnet_type, uq_method):
#     model = resnet18(pretrained=True, dropout_rate=0.5)


'''
    ResNetMCD

    This class embodies MC-sampling inside the forward. A simple backbone resnet takes a batch 
    of B images and outputs [B, C], where C is the number of predicted classes; for each image
    on the batch it returns the probability distribution, hence 10 p-values, one for each class.
    ResNetMCD, instead, embodies MC-sampling, hence it returns [S, B, C], where S is the
    Monte-Carlo sample size.
'''
import utils.constants as keys


class ResNetMCD(nn.Module):
    def __init__(self, resnet_type, dropout_rate=0.0, full_bayesian=False, pretrained=True, temperature=1.0,
                 weights=None, transform=None):
        super(ResNetMCD, self).__init__()

        # Guard for the ResNet type
        if resnet_type not in keys.SUPPORTED_RESNETS:
            raise Exception(f"{resnet_type} is not a supported ResNet.")

        self.semantic_segmentation_mode = resnet_type == 'resnet_fcn'

        # Using the pre-defined resnet as backbone architecture
        if resnet_type == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained, weights=weights, dropout_rate=dropout_rate,
                                     full_bayesian=full_bayesian)
        elif resnet_type == 'resnet34':
            self.backbone = resnet34(pretrained=pretrained, weights=weights, dropout_rate=dropout_rate,
                                     full_bayesian=full_bayesian)
        elif resnet_type == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained, weights=weights, dropout_rate=dropout_rate,
                                     full_bayesian=full_bayesian)
        elif resnet_type == 'resnet_fcn':
            weights = FCN_ResNet50_Weights.DEFAULT
            self.backbone = fcn_mcd_resnet50(weights=weights, dropout_rate=dropout_rate, full_bayesian=full_bayesian)

        elif resnet_type == "robust_resnet":
            self.backbone = weights

        # Setting up the temperature for temperature scaling
        self.temperature = temperature

        self.transform = transform

    # Returning a batch of predictions using MC-sampling
    def forward(self, x, mc_sample_size=100, get_mc_output=False):
        """
        get_mean: bool
        set to False to obtain output vector with shape (mc_sample_size, batch_size, n_classes)
        """

        # NOTE: This does not affect the FCN for now! Integrate it for implementing the attack
        # if normalization step is included in the model perform it before forward pass
        if self.transform is not None:
            x = self.transform(x)

        # print(f"A MC batch of {x.shape[0]} x {mc_sample_size} = {new_x.shape[0]} has been created. Make sure it fits your GPU!")

        # Repeating the input for obtaining a big monte-carlo batch
        new_x = x.repeat(mc_sample_size, 1, 1, 1)

        # Computing the output
        out = self.backbone(new_x)

        # If we are in Semantic Segmentation we extract the true out
        if self.semantic_segmentation_mode:
            out = out["out"]
            mc_batch, num_classes, h, w = out.shape
            out = out.view(mc_sample_size, mc_batch // mc_sample_size, num_classes, h, w)
        else:
            # Re-organizing the output
            mc_batch, num_classes = out.shape
            out = out.view(mc_sample_size, mc_batch // mc_sample_size, num_classes)

        if not get_mc_output:
            out = out.mean(dim=0)

        # Performing temperature scaling on the logits
        out = temperature_scaling(out, self.temperature)

        return out

    def eval(self, activate_dropout=True):
        """
        Overriding the classic eval() method to activate dropout as default at inference time
        """
        self.train(False)
        if activate_dropout:
            # Reactivating the dropout layers
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.training = True
        return self

    def check_if_dropout_is_active(self, return_counts=False):
        cnt, cnt_dropout, cnt_dropout_active = 0, 0, 0
        for module in self.modules():  # NOTE: Module is not defined. Fix it
            cnt += 1
            if isinstance(module, nn.Dropout):
                cnt_dropout += 1
                if module.training is True:
                    cnt_dropout_active += 1

        dropout_is_active = cnt_dropout_active > 0
        if return_counts:
            return dropout_is_active, cnt_dropout_active, cnt_dropout, cnt
        else:
            return dropout_is_active
        # return utils.check_if_dropout_is_active(self, return_counts=return_counts)


class ResNetEnsemble(nn.Module):
    def __init__(self, resnet_type, num_ensemble, transform=None):
        super(ResNetEnsemble, self).__init__()

        # Guard for the ResNet type
        if resnet_type not in keys.SUPPORTED_RESNETS:
            raise Exception(f"{resnet_type} is not a supported ResNet.")

        if num_ensemble > keys.MAXIMUM_ENSEMBLE_SIZE:
            raise Exception(
                f"Exceded the maximum ensemble size. Try using at most {keys.MAXIMUM_ENSEMBLE_SIZE} members.")

        # Setting up basic attributes
        self.temperature = 1
        self.num_ensemble = num_ensemble
        self.base_model_path = os.path.join('models', 'deep_ensemble', resnet_type)

        self.transform = transform

        # Using the pre-defined resnet as backbone architecture
        memebr_list = []
        for e_id in range(num_ensemble):
            member = None
            if resnet_type == 'resnet18':
                member = resnet18(pretrained=False, dropout_rate=0.0)
            elif resnet_type == 'resnet34':
                member = resnet34(pretrained=False, dropout_rate=0.0)
            elif resnet_type == 'resnet50':
                member = resnet50(pretrained=False, dropout_rate=0.0)

            member_path = os.path.join(self.base_model_path, f'model{e_id + 1}.pt')
            member.load_state_dict(torch.load(member_path))
            memebr_list.append(member)

        self.ensemble = nn.ModuleList(memebr_list)

        self.out_features = list(self.ensemble[0].children())[-1].out_features

    # Returning a batch of predictions using MC-sampling
    def forward(self, x, mc_sample_size=100, get_mc_output=False):
        # TODO: mc_sample_size porcheria per non rompere l'interfaccia, se si trova un altra soluzione meglio

        if self.transform is not None:
            x = self.transform(x)

        # Preparing the out tensor
        batch_size = x.shape[0]
        out = torch.zeros(size=(self.num_ensemble, batch_size, self.out_features)).to(x.device)

        # Computing the set of ensemble prediction logits
        for i, member in enumerate(self.ensemble):
            out[i] = member(x)

        # Obtaining the mean logit vector
        if not get_mc_output:
            out = out.mean(dim=0)

        # Performing temperature scaling on the logits
        out = temperature_scaling(out, self.temperature)

        return out


class ResNet_DUQ(nn.Module):
    def __init__(
            self,
            feature_extractor=torchvision.models.resnet18(),
            num_classes=10,
            centroid_size=512,
            model_output_size=512,
            length_scale=0.1,
            gamma=0.999,
            transform=None,
            device='cpu'
    ):
        super().__init__()

        self.gamma = gamma
        self.transform = transform

        self.W = nn.Parameter(
            torch.zeros(centroid_size, num_classes, model_output_size)
        )
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        # Forcing resnet18 with identity feature extractor
        self.feature_extractor = torchvision.models.resnet18()
        self.feature_extractor.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.feature_extractor.maxpool = torch.nn.Identity()
        self.feature_extractor.fc = torch.nn.Identity()

        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer(
            "m", torch.normal(torch.zeros(centroid_size, num_classes), 0.05)
        )
        self.m = self.m * self.N

        self.sigma = length_scale

        # Loading the model
        self.load_state_dict(
            torch.load(os.path.join('models', 'deterministic_uq', 'resnet18.pt'), map_location=torch.device(device)))

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        return diff

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        z = self.feature_extractor(x)

        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum

    def forward(self, x, mc_sample_size=0, get_mc_output=True):
        if self.transform is not None:
            x = self.transform(x)

        z = self.feature_extractor(x)
        y_pred = self.rbf(z)

        return y_pred
