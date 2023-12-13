import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
# import models.model_utils as model_utils

first_RUN = True

config_cp_field_shallow_m2 = {
    "name": "config_cp_field_shallow_m2",
    "input_shape": [8, 1, -1, -1],
    "n_classes": 7,
    "depth": 26,
    "base_channels": 128,
    "n_blocks_per_stage": [3, 1, 1],
    "stage1": {"maxpool": [1, 2], "k1s": [3, 3, 3], "k2s": [1, 3, 3]},
    "stage2": {"maxpool": [], "k1s": [3, ], "k2s": [3, ]},
    "stage3": {"maxpool": [], "k1s": [1, ], "k2s": [1, ]},
    "block_type": "basic"
}

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        if module.bias is not None:
            module.bias.data.zero_()

def calc_padding(kernel):
    try:
        return kernel // 3
    except TypeError:
        return [k // 3 for k in kernel]


def getpad(x):
    a = (torch.arange(x.size(2)).float() * 2 /
         x.size(2) - 1.).unsqueeze(1).expand(-1, x.size(3)).unsqueeze(0).unsqueeze(0).expand(x.size(0), -1, -1, -1)
    if torch.cuda.is_available():
        return a.cuda()
    else:
        return a


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, k1=3, k2=3):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels + 1,
            out_channels,
            kernel_size=k1,
            stride=stride,  # downsample with first conv
            padding=calc_padding(k1),
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels + 1,
            out_channels,
            kernel_size=k2,
            stride=1,
            padding=calc_padding(k2),
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,  # downsample
                                                       padding=0, bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        oldx = x
        x = torch.cat([x, getpad(x)], dim=1)
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # y = F.selu(self.bn1(self.conv1(x)), inplace=True)
        y = torch.cat([y, getpad(y)], dim=1)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(oldx)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        # y = F.selu(y, inplace=True)  # apply ReLU after addition
        return y


class CPResnet(nn.Module):
    def __init__(self, config=None, num_targets=None, pooling='avg', ff='conv'):
        super(CPResnet, self).__init__()
        assert config is not None, print("CPResnet config required!")
        self.config = config
        # print("CONFIG:")
        # print(self.config)

        self.num_targets = num_targets
        self.dataset_name = config.get("dataset_name")
        input_shape = config['input_shape']
        n_classes = self.num_targets if isinstance(num_targets, int) else self.num_targets[0]
        self.n_classes = n_classes

        base_channels = config['base_channels']
        block_type = config['block_type']
        self.block_type = block_type
        depth = config['depth']
        self.pooling_padding = config.get("pooling_padding", 0) or 0
        assert block_type == 'basic', print("Only basic block implemented!")

        block = BasicBlock
        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth

        n_blocks_per_stage = [n_blocks_per_stage, n_blocks_per_stage, n_blocks_per_stage]
        if config.get("n_blocks_per_stage") is not None:
            n_blocks_per_stage = config.get("n_blocks_per_stage")
        self.n_blocks_per_stage = n_blocks_per_stage

        n_channels = config.get("n_channels")
        if n_channels is None:
            n_channels = [
                base_channels,
                base_channels * 2 * block.expansion,
                base_channels * 4 * block.expansion
            ]

        self.n_channels = n_channels
        self.in_c = nn.Sequential(nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=5,
            stride=2,
            padding=1,
            bias=False),
            nn.BatchNorm2d(n_channels[0]),
            nn.ReLU(True)
            # nn.SELU(True)
        )
        self.stage1 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage[0], block, stride=1, maxpool=config['stage1']['maxpool'],
            dropout=config['stage1'].get('dropout', None), k1s=config['stage1']['k1s'], k2s=config['stage1']['k2s'])
        self.stage2 = self._make_stage(
            n_channels[0], n_channels[1], n_blocks_per_stage[1], block, stride=1, maxpool=config['stage2']['maxpool'],
            dropout=config['stage2'].get('dropout', None), k1s=config['stage2']['k1s'], k2s=config['stage2']['k2s'])
        self.stage3 = self._make_stage(
            n_channels[1], n_channels[2], n_blocks_per_stage[2], block, stride=1, maxpool=config['stage3']['maxpool'],
            dropout=config['stage3'].get('dropout', None), k1s=config['stage3']['k1s'], k2s=config['stage3']['k2s'])
        self.ff_list = []

        if pooling == 'avg':
            pooler = nn.AdaptiveAvgPool2d
        else:
            pooler = nn.AdaptiveMaxPool2d
        self.pool = pooler((1, 1))

        if ff is not None:
            if ff == 'conv':
                self.ff_list += [nn.Conv2d(
                    n_channels[2],
                    n_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                    nn.BatchNorm2d(n_classes),
                ]
                self.ff_list += [pooler((1, 1))]
                self.ff_list += [nn.Flatten()]
            else:
                self.ff_list += [pooler((1, 1))]
                self.ff_list += [nn.Flatten()]
                self.ff_list += [nn.Linear(
                    n_channels[2],
                    64,
                    bias=True),
                ]
                self.ff_list += [nn.Linear(
                    64,
                    n_classes,
                    bias=True),
                ]

            self.feed_forward = nn.Sequential(
                *self.ff_list
            )
        else:
            self.feed_forward = nn.Sequential()

        self.apply(self._initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride, maxpool=None, k1s=None, k2s=None, dropout=None):
        stage = nn.Sequential()
        if 0 in maxpool:
            stage.add_module("maxpool{}_{}".format(0, 0),
                             nn.MaxPool2d(2, 2, padding=self.pooling_padding))

        for index in range(n_blocks):
            stage.add_module('block{}'.format(index + 1),
                             block(in_channels,
                                   out_channels,
                                   stride=stride, k1=k1s[index], k2=k2s[index]))

            in_channels = out_channels
            stride = 1
            # if index + 1 in maxpool:
            for m_i, mp_pos in enumerate(maxpool):
                if index + 1 == mp_pos:
                    stage.add_module("maxpool{}_{}".format(index + 1, m_i),
                                     nn.MaxPool2d(2, 2, padding=self.pooling_padding))
                    # stage.add_module("maxpool{}_{}".format(index + 1, m_i),
                    #                  nn.AvgPool2d(2, 2, padding=self.pooling_padding))

            # for d_i, d_pos in enumerate(dropout):
            #     if index + 1 == d_pos:
            #         stage.add_module("dropout{}_{}".format(index + 1, d_i),
            #                          nn.Dropout2d(0.5))
        return stage

    def forward_conv(self, x):
        global first_RUN

        # if first_RUN: print("x:", x.size())
        x = self.in_c(x)
        # if first_RUN: print("in_c:", x.size())
        x = self.stage1(x)
        # if first_RUN: print("stage1:", x.size())
        x = self.stage2(x)
        # if first_RUN: print("stage2:", x.size())
        x = self.stage3(x)
        # if first_RUN: print("stage3:", x.size())
        return x

    def forward(self, x, **kwargs):
        global first_RUN
        x = self.forward_conv(x)
        # embedding = self.pool(x)
        embedding = x
        x = self.feed_forward(x)
        # if first_RUN: print("feed_forward:", x.size())
        logit = x
        # if first_RUN: print("logit:", logit.size())
        first_RUN = False
        return {"output": logit, "embedding": embedding}

    def get_config(self):
        return self.config

    @staticmethod
    def _initialize_weights(module):
        initialize_weights(module)

    @property
    def name(self):
        return f"CPResnet_{self.block_type}"


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, *args):
        x, alpha = args
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *args):
        grad_output = args[0]
        output = grad_output.neg() * ctx.alpha
        return output, None


class CPResnet_BackProp(CPResnet):
    def __init__(self, *args, **kwargs):
        super(CPResnet_BackProp, self).__init__(*args, **kwargs)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.discriminator = nn.Sequential(
            # GradientReversal(),
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x, **kwargs):
        global first_RUN
        num_labeled = kwargs.get('num_labeled')
        if num_labeled is None:
            num_labeled = x.shape[0]

        emb = self.forward_conv(x)
        emb_pooled = self.pool(emb)
        domain = self.discriminator(
            ReverseLayerF.apply(emb_pooled.view(emb_pooled.size(0), -1), kwargs.get('lambda_', 1.)))
        x = self.feed_forward(emb[:num_labeled])
        if first_RUN: print("feed_forward:", x.size())
        logit = x.squeeze()
        if first_RUN: print("logit:", logit.size())
        first_RUN = False
        return {"output": logit, "embedding": emb_pooled, "domain": domain}

    @property
    def name(self):
        return "CPResnet_BackProp"


if __name__=='__main__':
    # basic = BasicBlock(128, 128, stride=2)
    print(CPResnet(config=config_cp_field_shallow_m2, num_targets=7))
