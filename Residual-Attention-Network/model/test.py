import torch
from torch import nn
from basic_layers import ResidualBlock

from residual_attention_network import ResidualAttentionModel_448input
from attention_module import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3, AttentionModule_stage0


class ResidualUnit(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualUnit, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_channels,
            int(output_channels / 4),
            1,
            1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(int(output_channels / 4))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            int(output_channels / 4),
            int(output_channels / 4),
            3,
            stride,
            padding=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(int(output_channels / 4))
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            int(output_channels / 4),
            output_channels,
            1,
            1,
            bias=False
        )

        self.conv4 = nn.Conv2d(
            input_channels,
            output_channels,
            1,
            stride,
            bias=False
        )

    def forward(self, x):
        residual = x

        # First 1x1 convolutional layer in the bottleneck residual unit
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)

        # Middle 3x3 convolutional layer in the bottleneck residual unit
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        # Third 1x1 convolutional layer in the bottleneck residual unit
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)

        out += residual

        return out


# r_unit = ResidualUnit(1, 64);
# output = r_unit(data)
# print(output.shape)


class AttentionModule(nn.Module):
    def __init__(self, input_channels, output_channels, p=1, t=2, r=1):
        super(AttentionModule, self).__init__()

        # Preprocessing residual units. The amount of residual units here is dependent on the value of p
        self.pre_processing_residual_units = nn.Sequential(
            *[ResidualUnit(input_channels, output_channels) for _ in range(p)]
        )

        # The amount of residual units here is dependent on the value of t
        self.trunk_branch = nn.Sequential(
            *[ResidualUnit(input_channels, output_channels) for _ in range(t)]
        )

        # Down Sampling one
        self.soft_mask_1 = nn.MaxPool2d(2, stride=2)
        self.r_block_between_adj_pooling_layer = nn.Sequential(
            *[ResidualUnit(input_channels, output_channels) for _ in range(r)]
        )

        # Down Sampling two
        self.soft_mask_2 = nn.MaxPool2d(2, stride=2)
        self.r_block_between_adj_pooling_layer = nn.Sequential(
            *[ResidualUnit(input_channels, output_channels) for _ in range(r)]
        )

        # Up Sampling one
        self.up_sample = nn.Sequential(
            *(
                module for _ in range(r) for module in [
                    ResidualUnit(input_channels, output_channels),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                ]
            )
        )

        # Up Sampling two
        self.up_sample = nn.Sequential(
            *(
                module for _ in range(r) for module in [
                    ResidualUnit(input_channels, output_channels),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                ]
            )
        )

        self.conv1 = nn.Conv2d(input_channels, output_channels, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(input_channels, output_channels, 1, 1, bias=False)
        self.activation = nn.Sigmoid()

        # Preprocessing residual units. The amount of residual units here is dependent on the value of p
        self.pre_processing_residual_units = nn.Sequential(
            *[ResidualUnit(input_channels, output_channels) for _ in range(p)]
        )

    def forward(self, x):
        # Pre-processing
        out = self.pre_processing_residual_units(x)

        # Trunk branch
        out_trunk = self.trunk_branch(out)

        # Mask branch
        down_sampled_out1 = self.soft_mask_1(out)

        out_mask = self.r_block_between_adj_pooling_layer(down_sampled_out1)
        print("After first down sample: ", out_mask.shape)
        out_mask = self.up_sample(out_mask)
        print("After up sample: ", out_mask.shape)

        down_sampled_out2 = self.soft_mask_2(out_mask)
        print("After second down sample: ", down_sampled_out2.shape)
        out_mask = self.r_block_between_adj_pooling_layer(down_sampled_out2)


        out_mask_plus_down_sampled_out2 = down_sampled_out2 + out_mask
        out_mask = self.up_sample(out_mask_plus_down_sampled_out2)

        # out_mask_plus_down_sampled_out1 = None
        # if down_sampled_out1.shape != out_mask.shape:
        #     m = self.conv1(down_sampled_out1)
        #     out_mask_plus_down_sampled_out1 = m + out_mask
        # else:
        #     out_mask_plus_down_sampled_out1 = down_sampled_out1 + out_mask
        #
        # out_mask = self.up_sample(out_mask_plus_down_sampled_out1)

        out_mask = self.up_sample(out_mask)

        out_mask = self.conv1(out_mask)
        out_mask = self.conv2(out_mask)
        out_mask = self.activation(out_mask)

        # print("Output of trunk branch", out_trunk.shape)
        # print("Output of mask branch", out_mask.shape)

        # Perform the calculation (1 + M(x)) * T(x)
        out = (1 + out_mask) * out_trunk
        return out


class ResidualAttentionNetwork(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualAttentionNetwork, self).__init__()

        self.layer_before_the_attention_modules = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        )

        self.attention_module_1 = nn.Sequential(
            ResidualUnit(output_channels, output_channels),
            AttentionModule(output_channels, output_channels)
        )

        self.attention_module_2 = nn.Sequential(
            ResidualUnit(output_channels, output_channels),
            AttentionModule(output_channels, output_channels)
        )

        self.attention_module_3 = nn.Sequential(
            ResidualUnit(output_channels, output_channels),
            AttentionModule(output_channels, output_channels)
        )

        self.residuals = nn.Sequential(
            ResidualUnit(output_channels, output_channels),
            ResidualUnit(output_channels, output_channels),
            ResidualUnit(output_channels, output_channels)
        )

        self.average_pooling = nn.AvgPool2d(4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(output_channels, 10)

    def forward(self, x):
        out = self.layer_before_the_attention_modules(x)
        out = self.attention_module_1(out)
        out = self.attention_module_2(out)
        out = self.attention_module_3(out)
        out = self.residuals(out)
        out = self.average_pooling(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out


# a = AttentionModule(1, 64)
# b = ResidualAttentionNetwork(1, 64)
# b(data)


import torch.nn as nn

class ResidualAttentionModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ResidualAttentionModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Remove fixed-size max-pooling layer
        self.mpool1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((None, None))  # Adaptive pooling
        )
        self.residual_block0 = ResidualBlock(64, 128)
        self.attention_module0 = AttentionModule_stage0(128, 128)
        self.residual_block1 = ResidualBlock(128, 256, 2)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        # Remove fixed-size average pooling layer
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Adaptive pooling
        )
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block0(out)
        out = self.attention_module0(out)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



shape = (5, 3, 150, 2500)
data = torch.randn(shape)

model = ResidualAttentionNetwork()
output = model(data)
print(output.shape)
