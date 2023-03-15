from typing import Tuple
import torch as ch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class MyResNet(nn.Module):
    def _copy_fc_weights(self, weight, bias, target_layer):
        target_layer.weight.data[:1000] = weight
        target_layer.bias.data[:1000] = bias

    def __init__(self, mask=None, num_classes: int = 2,
                 feature_layer: str = "x4",
                 mask_layer: str = "x4",
                 add_dropout: bool = False,
                 drop_prob: float = 0.5,
                 multi_fc: bool = False,
                 pretrained_weights: bool = False,
                 resnet_type='resnet18',
                 train_on_embedding: bool = False) -> None:
        super(MyResNet, self).__init__()
        if resnet_type == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained_weights)
            assert(self.model.fc.weight.data.shape[1] == 512)

        elif resnet_type == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained_weights)
            assert(self.model.fc.weight.data.shape[1] == 512)

        # elif resnet_type == 'resnet50':
        #     self.model = models.resnet50(pretrained=pretrained_weights)

        # elif resnet_type == 'resnext50_32x4d':
        #     self.model = models.resnext50_32x4d(pretrained=pretrained_weights)

        if pretrained_weights:
            weight = self.model.fc.weight.data
            bias = self.model.fc.bias.data

        self.model.fc = ch.nn.Identity()

        self.add_dropout = add_dropout

        self.train_on_embedding = train_on_embedding

        if multi_fc:
            layers = []
            layers += [nn.Linear(512, 128), nn.ReLU(inplace=True)]
            if self.add_dropout:
                layers.append(nn.Dropout(drop_prob))
            layers += [nn.Linear(128, 64), nn.ReLU(inplace=True)]
            if self.add_dropout:
                layers.append(nn.Dropout(drop_prob))
            layers.append(nn.Linear(64, num_classes))
            self.fc = nn.Sequential(*layers)
        else:
            self.fc = nn.Linear(512, num_classes)
            if pretrained_weights and num_classes >= 1000:
                self._copy_fc_weights(weight, bias, self.fc)

        # Dropout on feature representation
        # if self.add_dropout:
        #     self.dropout = nn.Dropout(drop_prob)

        # Set mask, remember which layer it is for
        self.mask = mask
        self.mask_layer = mask_layer
        # Remember which intermediate layer is to be used
        self.feature_layer = feature_layer

        print('Mask: %s, Mask layer: %s, Feature layer: %s' %
              ("Not None" if mask is not None else 'None', mask_layer, feature_layer))

    def forward(self, x: ch.Tensor, conditional_mask: ch.Tensor = None) -> Tuple[ch.Tensor, ch.tensor]:
        if self.train_on_embedding:
            x3 = x
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x3 = self.model.layer3(x)

        if self.train_on_embedding and self.feature_layer == 'x4':
            x4 = x
        else:
            if self.mask_layer == "x3" and self.mask is not None:
                if conditional_mask is None:
                    x3 = x3 * self.mask  # hypothesize verfication
                else:
                    with ch.no_grad():
                        x3[conditional_mask].data = x3[conditional_mask].data * self.mask

            x4 = self.model.layer4(x3)
            x4 = self.model.avgpool(x4)
            x4 = ch.flatten(x4, 1)

        if self.mask_layer == "x4" and self.mask is not None:
            if conditional_mask is None:
                x4 = x4 * self.mask  # hypothesize verfication
            else:
                x4[conditional_mask] = x4[conditional_mask] * self.mask

        # if self.add_dropout:
        #     x4 = self.dropout(x4)

        x = self.fc(x4)

        if self.feature_layer == "x4":
            latent = x4
        elif self.feature_layer == "x3":
            latent = x3
        else:
            raise NotImplementedError("Requested intermediate layer not in model")

        return x, latent

class MyDenseNet(nn.Module):
    def __init__(self, mask=None, num_classes: int = 2,
                 feature_layer: str = "x4",
                 mask_layer: str = "x4",
                 add_dropout: bool = False,
                 drop_prob: float = 0.5,
                 multi_fc: bool = False,
                 pretrained_weights: bool = False) -> None:
        super(MyDenseNet, self).__init__()
        self.model = models.densenet121(pretrained=pretrained_weights)
        self.model.classifier = ch.nn.Identity()

        self.fc = nn.Linear(1024, num_classes)

        # Set mask, remember which layer it is for
        self.mask = mask
        self.mask_layer = mask_layer
        assert mask_layer == 'x4'
        # Remember which intermediate layer is to be used
        self.feature_layer = feature_layer
        assert feature_layer == 'x4'

        print('Mask: %s, Mask layer: %s, Feature layer: %s' %
              ("Not None" if mask is not None else 'None', mask_layer, feature_layer))

    def forward(self, x: ch.Tensor, conditional_mask: ch.Tensor = None) -> Tuple[ch.Tensor, ch.tensor]:
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        x4 = ch.flatten(out, 1)

        if self.mask is not None:
            x4 = x4 * self.mask  # hypothesize verfication

        out = self.fc(x4)
        return out, x4


class MyMobileNet(nn.Module):
    def __init__(self, mask=None, num_classes: int = 2,
                 feature_layer: str = "c0",
                 mask_layer: str = "c0",
                 pretrained_weights: bool = False,
                 dropout: float = 0.5,
                 train_on_embedding: bool = False,
                 silent: bool = False) -> None:
        super(MyMobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained_weights)

        self.model.classifier = ch.nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            # nn.Identity(),
            nn.Linear(self.model.last_channel, num_classes),
        )

        self.train_on_embedding = train_on_embedding

        self.mask = mask
        self.mask_layer = mask_layer
        self.feature_layer = feature_layer

        if not silent:
            print('Mask: %s, Mask layer: %s, Feature layer: %s' %
                  ("Not None" if mask is not None else 'None', mask_layer, feature_layer))

    def forward(self, x: ch.Tensor, conditional_mask=False) -> Tuple[ch.Tensor, ch.tensor]:
        return self._forward_impl(x)

    def _forward_impl(self, x):
        if not self.train_on_embedding:
            x = self.model.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = ch.flatten(x, 1)
        latent = x
        x = self.classifier(x)
        return x, latent


class NoiseModule(nn.Module):
    def __init__(self, num=32, for_fc=False) -> None:
        super(NoiseModule, self).__init__()
        if for_fc:
            noise_ = ch.randn(1, num)
        else:
            noise_ = ch.randn(1, num, 1)
        noise_ = F.normalize(noise_)
        self.noise = ch.nn.Parameter(noise_)

    def forward(self, x):
        return x
