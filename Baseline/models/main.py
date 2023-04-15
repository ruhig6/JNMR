import torch
import cupy_module.adacof as adacof
import sys
import torch.nn as nn
from torch.nn import functional as F
from utility import CharbonnierFunc, moduleNormalize


class AdaCoFNet(torch.nn.Module):
    def __init__(self, args):
        super(AdaCoFNet, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        self.dilation = args.dilation

        self.get_kernel = PrunedKernelEstimation(self.kernel_size)
        
        self.context_synthesis = GridNet(12, 3)  # (in_channel, out_channel) = (126, 3) for the synthesis network

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

    def forward(self, frame0, frame1, frame3, frame4):
        h1 = int(list(frame1.size())[2])
        w1 = int(list(frame1.size())[3])
        h3 = int(list(frame3.size())[2])
        w3 = int(list(frame3.size())[3])
        if h1 != h3 or w1 != w3:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h1 % 32 != 0:
            pad_h = 32 - (h1 % 32)
            frame0 = F.pad(frame0, [0, 0, 0, pad_h], mode='reflect')
            frame1 = F.pad(frame1, [0, 0, 0, pad_h], mode='reflect')
            frame3 = F.pad(frame3, [0, 0, 0, pad_h], mode='reflect')
            frame4 = F.pad(frame4, [0, 0, 0, pad_h], mode='reflect')
            h_padded = True

        if w1 % 32 != 0:
            pad_w = 32 - (w1 % 32)
            frame0 = F.pad(frame0, [0, pad_w, 0, 0], mode='reflect')
            frame1 = F.pad(frame1, [0, pad_w, 0, 0], mode='reflect')
            frame3 = F.pad(frame3, [0, pad_w, 0, 0], mode='reflect')
            frame4 = F.pad(frame4, [0, pad_w, 0, 0], mode='reflect')
            w_padded = True
            
        Weight0, Weight1, Weight3, Weight4, \
        Alpha0, Alpha1, Alpha3, Alpha4, \
        Beta0, Beta1, Beta3, Beta4, \
        Occlusion, Blend \
            = self.get_kernel(moduleNormalize(frame0),
                        moduleNormalize(frame1),
                        moduleNormalize(frame3),
                        moduleNormalize(frame4))

        tensorAdaCoF0 = self.moduleAdaCoF(self.modulePad(frame0), Weight0, Alpha0, Beta0, self.dilation) * 1.
        tensorAdaCoF1 = self.moduleAdaCoF(self.modulePad(frame1), Weight1, Alpha1, Beta1, self.dilation) * 1.
        tensorAdaCoF3 = self.moduleAdaCoF(self.modulePad(frame3), Weight3, Alpha3, Beta3, self.dilation) * 1.
        tensorAdaCoF4 = self.moduleAdaCoF(self.modulePad(frame4), Weight4, Alpha4, Beta4, self.dilation) * 1.
        
        w, h = self.modulePad(frame1).shape[2:]
        
        tensorCombined = torch.cat(
            [tensorAdaCoF0, tensorAdaCoF1, tensorAdaCoF3, tensorAdaCoF4], dim=1)
        
        frame2_warp = Occlusion * tensorAdaCoF0 + Occlusion * tensorAdaCoF1  + (1 - Occlusion) * tensorAdaCoF3 + (1 - Occlusion) * tensorAdaCoF4 
        frame2_feat = self.context_synthesis(tensorCombined)

        frame2 = Blend * frame2_feat + (1 - Blend) * frame2_warp  # blending of the feature warp and ordinary warp


        
        if h_padded:
            frame2 = frame2[:, :, 0:h1, :]
        if w_padded:
            frame2 = frame2[:, :, :, 0:w1]

        if self.training:
            # Smoothness Terms
            m_Alpha0 = torch.mean(Weight0 * Alpha0, dim=1, keepdim=True)
            m_Alpha1 = torch.mean(Weight1 * Alpha1, dim=1, keepdim=True)
            m_Alpha3 = torch.mean(Weight3 * Alpha3, dim=1, keepdim=True)
            m_Alpha4 = torch.mean(Weight4 * Alpha4, dim=1, keepdim=True)

            m_Beta0 = torch.mean(Weight0 * Beta0, dim=1, keepdim=True)
            m_Beta1 = torch.mean(Weight1 * Beta1, dim=1, keepdim=True)
            m_Beta3 = torch.mean(Weight3 * Beta3, dim=1, keepdim=True)
            m_Beta4 = torch.mean(Weight4 * Beta4, dim=1, keepdim=True)

            g_Alpha0 = CharbonnierFunc(m_Alpha0[:, :, :, :-1] - m_Alpha0[:, :, :, 1:]) + CharbonnierFunc(
                m_Alpha0[:, :, :-1, :] - m_Alpha0[:, :, 1:, :])
            g_Beta0 = CharbonnierFunc(m_Beta0[:, :, :, :-1] - m_Beta0[:, :, :, 1:]) + CharbonnierFunc(
                m_Beta0[:, :, :-1, :] - m_Beta0[:, :, 1:, :])

            g_Alpha1 = CharbonnierFunc(m_Alpha1[:, :, :, :-1] - m_Alpha1[:, :, :, 1:]) + CharbonnierFunc(
                m_Alpha1[:, :, :-1, :] - m_Alpha1[:, :, 1:, :])
            g_Beta1 = CharbonnierFunc(m_Beta1[:, :, :, :-1] - m_Beta1[:, :, :, 1:]) + CharbonnierFunc(
                m_Beta1[:, :, :-1, :] - m_Beta1[:, :, 1:, :])

            g_Alpha3 = CharbonnierFunc(m_Alpha3[:, :, :, :-1] - m_Alpha3[:, :, :, 1:]) + CharbonnierFunc(
                m_Alpha3[:, :, :-1, :] - m_Alpha3[:, :, 1:, :])
            g_Beta3 = CharbonnierFunc(m_Beta3[:, :, :, :-1] - m_Beta3[:, :, :, 1:]) + CharbonnierFunc(
                m_Beta3[:, :, :-1, :] - m_Beta3[:, :, 1:, :])

            g_Alpha4 = CharbonnierFunc(m_Alpha4[:, :, :, :-1] - m_Alpha4[:, :, :, 1:]) + CharbonnierFunc(
                m_Alpha4[:, :, :-1, :] - m_Alpha4[:, :, 1:, :])
            g_Beta4 = CharbonnierFunc(m_Beta4[:, :, :, :-1] - m_Beta4[:, :, :, 1:]) + CharbonnierFunc(
                m_Beta4[:, :, :-1, :] - m_Beta4[:, :, 1:, :])
            
            g_Occlusion = CharbonnierFunc(Occlusion[:, :, :, :-1] - Occlusion[:, :, :, 1:]) + CharbonnierFunc(
                Occlusion[:, :, :-1, :] - Occlusion[:, :, 1:, :])

            g_Spatial = g_Alpha0 + g_Beta0 + g_Alpha1 + g_Beta1 + g_Alpha3 + g_Beta3 + g_Alpha4 + g_Beta4 

            return {'frame2': frame2, 'g_Spatial': g_Spatial, 'g_Occlusion': g_Occlusion}
        else:
            return frame2

        
class GridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96)):
        super(GridNet, self).__init__()

        self.n_row = 3
        self.n_col = 6
        self.n_chs = grid_chs
        assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'

        self.lateral_init = LateralBlock(in_chs, self.n_chs[0])

        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col - 1):
                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)

        return self.lateral_final(state_05)


class LateralBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(LateralBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
        return fx + x


class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DownSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.f(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.f(x)


class PrunedKernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(PrunedKernelEstimation, self).__init__()
        self.kernel_size = kernel_size
        
        self.Downsample = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.moduleConv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=48, out_channels=51, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=99, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=99, out_channels=97, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=97, out_channels=94, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=94, out_channels=156, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=156, out_channels=142, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=142, out_channels=159, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=159, out_channels=92, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=92, out_channels=72, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=72, out_channels=121, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=121, out_channels=99, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=99, out_channels=69, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=69, out_channels=36, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.moduleUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=36, out_channels=121, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=121, out_channels=74, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=74, out_channels=83, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=83, out_channels=81, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.moduleUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=81, out_channels=159, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=159, out_channels=83, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=83, out_channels=88, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=88, out_channels=72, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=72, out_channels=94, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=94, out_channels=45, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=45, out_channels=47, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=47, out_channels=44, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=44, out_channels=51, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        # forward
        self.moduleWeight0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=51, out_channels=49, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=21, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1),
            torch.nn.Softmax(dim=1)
        )
        self.moduleAlpha0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=50, out_channels=48, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=48, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=20, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1)
        )
        self.moduleBeta0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=50, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=20, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1)
        )
        self.moduleWeight1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=50, out_channels=49, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=49, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=20, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1),
            torch.nn.Softmax(dim=1)
        )
        self.moduleAlpha1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=50, out_channels=49, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=49, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=20, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1)
        )
        self.moduleBeta1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=50, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=20, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1)
        )

        # backward
        self.moduleWeight3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=50, out_channels=49, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=49, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=20, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1),
            torch.nn.Softmax(dim=1)
        )
        self.moduleAlpha3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=50, out_channels=49, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=49, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=20, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1)
        )
        self.moduleBeta3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=50, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=20, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1)
        )
        
        self.moduleWeight4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=50, out_channels=49, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=49, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=20, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1),
            torch.nn.Softmax(dim=1)
        )
        self.moduleAlpha4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=50, out_channels=49, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=49, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=20, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1)
        )
        self.moduleBeta4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=50, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=20, out_channels=self.kernel_size ** 2, kernel_size=3, stride=1, padding=1)
        )
        


        self.moduleOcclusion = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=52, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=52, out_channels=51, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=51, out_channels=48, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()
        )
        self.moduleBlend = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=51, out_channels=52, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=52, out_channels=51, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=51, out_channels=48, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, rfield0, rfield1, rfield3, rfield4):
    
        tensorJoin = torch.cat([rfield0, rfield1, rfield3, rfield4], 1)


        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine_5 = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine_5)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine_4 = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine_4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine_3 = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine_3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2
        
        
        # forward
        Weight0 = self.moduleWeight0(tensorCombine)
        Alpha0 = self.moduleAlpha0(tensorCombine)
        Beta0 = self.moduleBeta0(tensorCombine)
        Weight1 = self.moduleWeight1(tensorCombine)
        Alpha1 = self.moduleAlpha1(tensorCombine)
        Beta1 = self.moduleBeta1(tensorCombine)
        
        # backward
        Weight3 = self.moduleWeight3(tensorCombine)
        Alpha3 = self.moduleAlpha3(tensorCombine)
        Beta3 = self.moduleBeta3(tensorCombine)
        Weight4 = self.moduleWeight4(tensorCombine)
        Alpha4 = self.moduleAlpha4(tensorCombine)
        Beta4 = self.moduleBeta4(tensorCombine)
        
        Occlusion = self.moduleOcclusion(tensorCombine)
        Blend = self.moduleBlend(tensorCombine)

        return Weight0, Weight1, Weight3, Weight4, \
        Alpha0, Alpha1, Alpha3, Alpha4, \
        Beta0, Beta1, Beta3, Beta4, \
        Occlusion, Blend