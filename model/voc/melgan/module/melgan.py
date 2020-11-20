import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
import numpy as np
import scipy
from librosa.output import write_wav as write


class ModelLoss(nn.Module):
    def __init__(self, conf):
        torch.nn.Module.__init__(self)

        # reconstruction loss
        self.feat_match = conf['model']['feat_match']
        self.n_layers_D = conf['model']['d']['n_layers_D']
        self.num_D = conf['model']['d']['num_D']

    def forward(self, step, d_fake_out=None, d_real_out=None):
        if step == 'g':
            loss_G = 0
            for scale in d_fake_out:
                loss_G += -scale[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (self.n_layers_D + 1)
            D_weights = 1.0 / self.num_D
            wt = D_weights * feat_weights
            for i in range(self.num_D):
                for j in range(len(d_fake_out[i]) - 1):
                    loss_feat += wt * F.l1_loss(d_fake_out[i][j], d_real_out[i][j].detach())

            loss = loss_G + loss_feat * self.feat_match
            report_loss_keys = [
                {"loss_G": loss_G},
                {"loss_feat": loss_feat},
                {"loss": loss}]

            return loss, report_loss_keys

        elif step == 'd':
            loss_D = 0
            for scale in d_fake_out:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in d_real_out:
                loss_D += F.relu(1 - scale[-1]).mean()

            report_loss_keys = [
                {"loss_D": loss_D},
            ]
            return loss_D, report_loss_keys


class Model(nn.Module):
    def __init__(self, conf, is_training=True):
        super(Model, self).__init__()

        self.is_training = is_training
        self.sampling_rate = conf['data']['sampling_rate']
        model_g_conf = conf['model']['g']
        model_d_conf = conf['model']['d']
        self.netG = Generator(**model_g_conf)
        if self.is_training:
            self.netD = Discriminator(**model_d_conf)

        self.MelGANLoss = ModelLoss(conf)
        self.loss_snapshot_step = conf['train']['loss_snapshot_step']

    def forward(self, step=None, batch=None, logger=None, gs=None, valid=False, valid_num=None, device=None, outdir=None, pred_mels=None):
        if self.is_training:
            return self._forward(step, batch, logger, gs, valid, valid_num, device, outdir)
        else:
            return self._inference(pred_mels, device)

    def _forward(self, step, batch, logger, gs, valid=False, valid_num=None, device=None, outdir=None):
        mels = batch['mels_seg'].to(device)
        wavs = batch['audios_seg'].to(device)
        wavs_pred = self.netG(mels.detach())
        report_loss_keys = []
        if not valid:
            if step == 'g':
                d_fake_out = self.netD(wavs_pred)
                d_real_out = self.netD(wavs)
                loss, _report_loss_keys = self.MelGANLoss(step=step, d_fake_out=d_fake_out, d_real_out=d_real_out)
                report_loss_keys += _report_loss_keys

            elif step == 'd':
                d_fake_out = self.netD(wavs_pred.detach())
                d_real_out = self.netD(wavs)
                loss, _report_loss_keys = self.MelGANLoss(step=step, d_fake_out=d_fake_out, d_real_out=d_real_out)
                report_loss_keys += _report_loss_keys

            if logger is not None and not valid:
                if gs % int(self.loss_snapshot_step) == 0:
                    logger.log_loss(report_loss_keys, gs)

            return loss, report_loss_keys

        else:
            audio_real = wavs.squeeze(1)
            audio_real = audio_real.cpu().detach().numpy()
            audio_pred = wavs_pred.squeeze(1)
            audio_pred = audio_pred.cpu().detach().numpy()

            report_key_wavs = [
                {"audio_real_{}".format(valid_num): audio_real},
                {"audio_pred_{}".format(valid_num): audio_pred},
            ]

            logger.log_wav(report_key_wavs, gs)
            for report_wav in report_key_wavs:
                for k in report_wav.keys():
                    filename = "{}/{}_{}.step.wav".format(outdir, k, str(gs))
                    _audio = (report_wav[k][0] * 32768.0).astype("int16")
                    scipy.io.wavfile.write(filename, self.sampling_rate, _audio)

    def _inference(self, pred_mels, device):
        wavs_pred = self.netG(pred_mels.to(device).detach())
        audio_pred = wavs_pred.squeeze(1)
        audio_pred = audio_pred.cpu().detach().numpy()

        return audio_pred

def optimizer(conf, melgan):
    conf_opt = conf['optimizer']
    optimizer_g = torch.optim.Adam(
        melgan.netG.parameters(),
        lr=float(conf_opt['adam_alpha']),
        betas=(float(conf_opt['adam_beta1']), float(conf_opt['adam_beta2'])))
    optimizer_d = torch.optim.Adam(
        melgan.netD.parameters(),
        lr=float(conf_opt['adam_alpha']),
        betas=(float(conf_opt['adam_beta1']), float(conf_opt['adam_beta2'])))

    optimizers = {"optimizer_g": optimizer_g,
                  "optimizer_d": optimizer_d}

    return optimizers


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Generator(nn.Module):
    def __init__(self, mel_channels, ngf, n_residual_layers):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(mel_channels, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers_D, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers_D + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers_D + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers_D + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, num_D, ndf, n_layers_D, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers_D, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results