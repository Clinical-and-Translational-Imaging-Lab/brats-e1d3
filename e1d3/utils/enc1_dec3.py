import torch
from torch import nn


class PrototypeArchitecture3d(nn.Module):
    """Draft of the proposed 1Enc-3Dec Architecture
    Each decoder predict one of WT, TC, EN
    No coupling between decoders

    Input-Shape: 4x96x96x96
    Output-Shape: {2x96x96x96}x3 (softmax)

    > PathA: WT, PathB: TC, PathC: EN
    """

    def __init__(self, config=None):
        super(PrototypeArchitecture3d, self).__init__()
        if config is not None:
            config_data = config['data']
            num_channels = len(config_data.get('channels'))
        else:
            num_channels = 4

        ######################### Encoder:
        # Level 1
        self.conv_enc_1_1 = Conv3dNormAct(num_channels, 30, kernel_size=3, padding=1)
        self.conv_enc_1_2 = Conv3dNormAct(30, 30, kernel_size=3, padding=1)

        # Level 2
        self.downsampleby2_2 = nn.Conv3d(30, 30, kernel_size=3, stride=2, padding=1)
        self.conv_enc_2_1 = Conv3dNormAct(30, 60, kernel_size=3, padding=1)
        self.conv_enc_2_2 = Conv3dNormAct(60, 60, kernel_size=3, padding=1)

        # Level 3
        self.downsampleby2_3 = nn.Conv3d(60, 60, kernel_size=3, stride=2, padding=1)
        self.conv_enc_3_1 = Conv3dNormAct(60, 120, kernel_size=3, padding=1)
        self.conv_enc_3_2 = Conv3dNormAct(120, 120, kernel_size=3, padding=1)

        # Level 4
        self.downsampleby2_4 = nn.Conv3d(120, 120, kernel_size=3, stride=2, padding=1)
        self.conv_enc_4_1 = Conv3dNormAct(120, 240, kernel_size=3, padding=1)
        self.conv_enc_4_2 = Conv3dNormAct(240, 240, kernel_size=3, padding=1)

        # Level 5
        self.downsampleby2_5 = nn.Conv3d(240, 240, kernel_size=3, stride=2, padding=1)
        self.conv_enc_5 = Conv3dNormAct(240, 480, kernel_size=3, padding=1)

        ######################### Decoder:

        ############### Path A
        # Level 5
        self.conv_dec_5_pathA = Conv3dNormAct(480, 240, kernel_size=3, padding=1)

        # Level 4
        self.upsample_4_pathA = nn.ConvTranspose3d(240, 240, kernel_size=2, stride=2)
        self.conv_dec_4_pathA_1 = Conv3dNormAct(240 + 240, 240, kernel_size=3, padding=1)
        self.conv_dec_4_pathA_2 = Conv3dNormAct(240, 120, kernel_size=3, padding=1)

        # Level 3
        self.upsample_3_pathA = nn.ConvTranspose3d(120, 120, kernel_size=2, stride=2)
        self.conv_dec_3_pathA_1 = Conv3dNormAct(120 + 120, 120, kernel_size=3, padding=1)
        self.conv_dec_3_pathA_2 = Conv3dNormAct(120, 60, kernel_size=3, padding=1)

        # Level 2
        self.upsample_2_pathA = nn.ConvTranspose3d(60, 60, kernel_size=2, stride=2)
        self.conv_dec_2_pathA_1 = Conv3dNormAct(60 + 60, 60, kernel_size=3, padding=1)
        self.conv_dec_2_pathA_2 = Conv3dNormAct(60, 30, kernel_size=3, padding=1)

        # Level 1
        self.upsample_1_pathA = nn.ConvTranspose3d(30, 30, kernel_size=2, stride=2)
        self.conv_dec_1_pathA_1 = Conv3dNormAct(30 + 30, 30, kernel_size=3, padding=1)
        self.conv_dec_1_pathA_2 = Conv3dNormAct(30, 15, kernel_size=3, padding=1)

        self.conv_pathA = nn.Conv3d(15, 2, kernel_size=1)

        ############### Path B
        # Level 5
        self.conv_dec_5_pathB = Conv3dNormAct(480, 240, kernel_size=3, padding=1)

        # Level 4
        self.upsample_4_pathB = nn.ConvTranspose3d(240, 240, kernel_size=2, stride=2)
        self.conv_dec_4_pathB_1 = Conv3dNormAct(240 + 240, 240, kernel_size=3, padding=1)
        self.conv_dec_4_pathB_2 = Conv3dNormAct(240, 120, kernel_size=3, padding=1)

        # Level 3
        self.upsample_3_pathB = nn.ConvTranspose3d(120, 120, kernel_size=2, stride=2)
        self.conv_dec_3_pathB_1 = Conv3dNormAct(120 + 120, 120, kernel_size=3, padding=1)
        self.conv_dec_3_pathB_2 = Conv3dNormAct(120, 60, kernel_size=3, padding=1)

        # Level 2
        self.upsample_2_pathB = nn.ConvTranspose3d(60, 60, kernel_size=2, stride=2)
        self.conv_dec_2_pathB_1 = Conv3dNormAct(60 + 60, 60, kernel_size=3, padding=1)
        self.conv_dec_2_pathB_2 = Conv3dNormAct(60, 30, kernel_size=3, padding=1)

        # Level 1
        self.upsample_1_pathB = nn.ConvTranspose3d(30, 30, kernel_size=2, stride=2)
        self.conv_dec_1_pathB_1 = Conv3dNormAct(30 + 30, 30, kernel_size=3, padding=1)
        self.conv_dec_1_pathB_2 = Conv3dNormAct(30, 15, kernel_size=3, padding=1)

        self.conv_pathB = nn.Conv3d(15, 2, kernel_size=1)

        ############### Path C
        # Level 5
        self.conv_dec_5_pathC = Conv3dNormAct(480, 240, kernel_size=3, padding=1)

        # Level 4
        self.upsample_4_pathC = nn.ConvTranspose3d(240, 240, kernel_size=2, stride=2)
        self.conv_dec_4_pathC_1 = Conv3dNormAct(240 + 240, 240, kernel_size=3, padding=1)
        self.conv_dec_4_pathC_2 = Conv3dNormAct(240, 120, kernel_size=3, padding=1)

        # Level 3
        self.upsample_3_pathC = nn.ConvTranspose3d(120, 120, kernel_size=2, stride=2)
        self.conv_dec_3_pathC_1 = Conv3dNormAct(120 + 120, 120, kernel_size=3, padding=1)
        self.conv_dec_3_pathC_2 = Conv3dNormAct(120, 60, kernel_size=3, padding=1)

        # Level 2
        self.upsample_2_pathC = nn.ConvTranspose3d(60, 60, kernel_size=2, stride=2)
        self.conv_dec_2_pathC_1 = Conv3dNormAct(60 + 60, 60, kernel_size=3, padding=1)
        self.conv_dec_2_pathC_2 = Conv3dNormAct(60, 30, kernel_size=3, padding=1)

        # Level 1
        self.upsample_1_pathC = nn.ConvTranspose3d(30, 30, kernel_size=2, stride=2)
        self.conv_dec_1_pathC_1 = Conv3dNormAct(30 + 30, 30, kernel_size=3, padding=1)
        self.conv_dec_1_pathC_2 = Conv3dNormAct(30, 15, kernel_size=3, padding=1)

        self.conv_pathC = nn.Conv3d(15, 2, kernel_size=1)

    def forward(self, x_1):
        """"""

        ######################### Encoder:
        # Level 1
        x_1 = self.conv_enc_1_1(x_1)
        x_1 = self.conv_enc_1_2(x_1)

        # Level 2
        x_2 = self.downsampleby2_2(x_1)
        x_2 = self.conv_enc_2_1(x_2)
        x_2 = self.conv_enc_2_2(x_2)

        # Level 3
        x_3 = self.downsampleby2_3(x_2)
        x_3 = self.conv_enc_3_1(x_3)
        x_3 = self.conv_enc_3_2(x_3)

        # Level 4
        x_4 = self.downsampleby2_4(x_3)
        x_4 = self.conv_enc_4_1(x_4)
        x_4 = self.conv_enc_4_2(x_4)

        # Level 5
        x_5 = self.downsampleby2_5(x_4)
        x_5 = self.conv_enc_5(x_5)

        ######################### Decoder:

        ############### Level 5
        x_pathA = self.conv_dec_5_pathA(x_5)
        x_pathB = self.conv_dec_5_pathB(x_5)
        x_pathC = self.conv_dec_5_pathC(x_5)

        ############### Level 4
        # Upsampling
        x_pathA = self.upsample_4_pathA(x_pathA)  # Path A
        x_pathB = self.upsample_4_pathB(x_pathB)  # Path B
        x_pathC = self.upsample_4_pathC(x_pathC)  # Path C
        # Concatenation & Processing
        x_pathA = self.conv_dec_4_pathA_2(
            self.conv_dec_4_pathA_1(
                torch.cat(
                    (x_pathA, x_4)
                    , dim=1)
            )
        )  # Path A
        x_pathB = self.conv_dec_4_pathB_2(
            self.conv_dec_4_pathB_1(
                torch.cat(
                    (x_pathB, x_4)
                    , dim=1)
            )
        )  # Path B
        x_pathC = self.conv_dec_4_pathC_2(
            self.conv_dec_4_pathC_1(
                torch.cat(
                    (x_pathC, x_4)
                    , dim=1)
            )
        )  # Path C

        ############### Level 3
        # Upsampling
        x_pathA = self.upsample_3_pathA(x_pathA)  # Path A
        x_pathB = self.upsample_3_pathB(x_pathB)  # Path B
        x_pathC = self.upsample_3_pathC(x_pathC)  # Path C
        # Concatenation & Processing
        x_pathA = self.conv_dec_3_pathA_2(
            self.conv_dec_3_pathA_1(
                torch.cat(
                    (x_pathA, x_3)
                    , dim=1)
            )
        )  # Path A
        x_pathB = self.conv_dec_3_pathB_2(
            self.conv_dec_3_pathB_1(
                torch.cat(
                    (x_pathB, x_3)
                    , dim=1)
            )
        )  # Path B
        x_pathC = self.conv_dec_3_pathC_2(
            self.conv_dec_3_pathC_1(
                torch.cat(
                    (x_pathC, x_3)
                    , dim=1)
            )
        )  # Path C

        ############### Level 2
        # Upsampling
        x_pathA = self.upsample_2_pathA(x_pathA)  # Path A
        x_pathB = self.upsample_2_pathB(x_pathB)  # Path B
        x_pathC = self.upsample_2_pathC(x_pathC)  # Path C
        # Concatenation & Processing
        x_pathA = self.conv_dec_2_pathA_2(
            self.conv_dec_2_pathA_1(
                torch.cat(
                    (x_pathA, x_2)
                    , dim=1)
            )
        )  # Path A
        x_pathB = self.conv_dec_2_pathB_2(
            self.conv_dec_2_pathB_1(
                torch.cat(
                    (x_pathB, x_2)
                    , dim=1)
            )
        )  # Path B
        x_pathC = self.conv_dec_2_pathC_2(
            self.conv_dec_2_pathC_1(
                torch.cat(
                    (x_pathC, x_2)
                    , dim=1)
            )
        )  # Path C

        ############### Level 1
        # Upsampling
        x_pathA = self.upsample_1_pathA(x_pathA)  # Path A
        x_pathB = self.upsample_1_pathB(x_pathB)  # Path B
        x_pathC = self.upsample_1_pathC(x_pathC)  # Path C
        # Concatenation & Processing
        x_pathA = self.conv_dec_1_pathA_2(
            self.conv_dec_1_pathA_1(
                torch.cat(
                    (x_pathA, x_1)
                    , dim=1)
            )
        )  # Path A
        x_pathB = self.conv_dec_1_pathB_2(
            self.conv_dec_1_pathB_1(
                torch.cat(
                    (x_pathB, x_1)
                    , dim=1)
            )
        )  # Path B
        x_pathC = self.conv_dec_1_pathC_2(
            self.conv_dec_1_pathC_1(
                torch.cat(
                    (x_pathC, x_1)
                    , dim=1)
            )
        )  # Path C

        # Classification
        x_pathA = self.conv_pathA(x_pathA)
        x_pathB = self.conv_pathB(x_pathB)
        x_pathC = self.conv_pathC(x_pathC)

        return x_pathA, x_pathB, x_pathC

    def print_model_parameters(self):
        """Helper to print out model parameters"""
        for param_tensor in self.state_dict():
            print(param_tensor, '\t', self.state_dict()[param_tensor].size())
        print("Total Parameters:", sum(param.numel() for param in self.parameters()))


class Conv3dNormAct(nn.Module):
    """Convolution3d -> Norm3d -> Activation"""

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                 dilation=(1, 1, 1)):
        super(Conv3dNormAct, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
        self.norm = torch.nn.InstanceNorm3d(num_features=out_channels, affine=True)
        self.acti = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        """"""
        return self.acti(self.norm(self.conv(x)))


if __name__ == '__main__':
    net = PrototypeArchitecture3d().cuda().train()
    net.print_model_parameters()

    x = torch.randn(3, 4, 96, 96, 96).cuda()
    ya, yb, yc = net(x)
    print(ya.shape, yb.shape, yc.shape)
    loss = ya.sum() + yb.sum() + yc.sum()
    loss.backward()
