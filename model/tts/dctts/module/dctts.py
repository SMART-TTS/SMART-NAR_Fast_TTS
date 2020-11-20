import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from model.tts.dctts.module.conv import *

class ModelLoss(nn.Module):
    def __init__(self, conf):
        torch.nn.Module.__init__(self)

        # reconstruction loss
        self.criterion_recon = nn.L1Loss()
        self.criterion_attn = nn.L1Loss()
        self.criterion_recon_bce = nn.BCEWithLogitsLoss()
        self.train_module = conf['train']['module']

    def forward(self, ys_pred, ys, attn_ws=None, attn_guide=None, attn_mask=None, dynamic_guide=None):
        if self.train_module == 't2m':
            ys_pred_logit = ys_pred[0]
            ys_pred = ys_pred[1]
            loss_recon = self.criterion_recon(ys_pred, ys)
            loss_recon_bce = self.criterion_recon_bce(ys_pred_logit, ys)
            attn_mask = torch.ones_like(attn_ws)
            loss_attn = self.criterion_attn(
                attn_guide * attn_ws * attn_mask,
                torch.zeros_like(attn_ws)) * dynamic_guide

            loss = loss_recon + loss_recon_bce + loss_attn
            report_loss_keys = [
                {"loss_recon": loss_recon},
                {"loss_recon_l1": loss_recon},
                {"loss_recon_bce": loss_recon_bce},
                {"loss_attn": loss_attn},
                {"loss": loss}]

        else:
            loss_recon = self.criterion_recon(ys_pred[1][:, :, :ys.size(2)], ys)
            loss_recon_bce = self.criterion_recon_bce(ys_pred[0][:, :, :ys.size(2)], ys)
            loss = loss_recon + loss_recon_bce
            report_loss_keys = [
                {"loss_recon": loss_recon},
                {"loss_recon_l1": loss_recon},
                {"loss_recon_bce": loss_recon_bce},
                {"loss": loss}]

        return loss, report_loss_keys


class Model(nn.Module):
    def __init__(self, conf, is_training=True):
        super(Model, self).__init__()

        self.is_training = is_training
        self.DCTTSLoss = ModelLoss(conf)
        self.loss_snapshot_step = conf['train']['loss_snapshot_step']
        self.max_target_len = conf['data']['data_max_mel_length']

        self.conf_model = conf['model']
        self.idim = self.conf_model['idim']
        self.fdim = self.conf_model['fdim']
        self.edim = self.conf_model['edim']
        self.ddim = self.conf_model['ddim']
        self.dropout_rate = self.conf_model['dropout_rate']

        self.train_module = conf['train']['module']
        if conf['train']['module'] == 't2m':
            self.text_encoder = TextEncoder(self.idim, self.edim, self.ddim, self.dropout_rate)
            self.audio_encoder = AudioEncoder(self.fdim, self.ddim, self.dropout_rate, is_training=is_training)
            self.audio_decoder = AudioDecoder(self.fdim, self.ddim, self.dropout_rate)

        else: # ssrn
            self.ssrn = SuperRes(self.fdim, self.ddim, self.dropout_rate)

    def forward(self, pred_mels=None, batch=None, valid=False, dynamic_guide=None, logger=None, gs=None, device=None, valid_num=None, report_name_for_outs=None, text=None):
        if self.is_training:
            return self._forward(batch, valid, dynamic_guide, logger, gs, device, valid_num, report_name_for_outs)
        else:
            return self._inference(text, pred_mels, batch, device)

    def _forward(self, batch, valid=False, dynamic_guide=None, logger=None, gs=None, device=None, valid_num=None, report_name_for_outs=None):

        texts = batch['text'].detach().to(device)
        mels = batch['mel'].detach().to(device)
        coarse_mels = batch['coarse_mel'].detach().to(device)
        attn_masks = batch['attn_mask'].detach().to(device)
        attn_guides = batch['attn_guide'].detach().to(device)

        if self.train_module == 't2m':
            shift_coarse_mels = torch.cat((torch.zeros((coarse_mels.size(0), coarse_mels.size(1), 1)).to(device), coarse_mels[:, :, :-1]), 2)

            ks, vs = self.text_encoder(texts)
            zs, attn_ws = self.audio_encoder(ks, vs, shift_coarse_mels)
            mels_pred = self.audio_decoder(zs)

            loss, report_keys_loss = self.DCTTSLoss(mels_pred, coarse_mels, attn_ws, attn_guides, attn_masks, dynamic_guide)

            if logger is not None:
                if not valid:
                    if gs % int(self.loss_snapshot_step) == 0:
                        logger.log_loss(report_keys_loss, gs)

                else:
                    ilens = batch['ilens'].detach().to(device)
                    coarse_olens = batch['coarse_olens'].detach().to(device)

                    report_mels_keys = [
                        {"mel_pred": mels_pred[1].transpose(1, 2)},
                        {"true_mel": coarse_mels.transpose(1, 2)},
                        ]

                    report_attn_ws_keys = [
                        {"attn_ws": attn_ws},
                        {"attn_guide": attn_guides},
                        {"attn_mask": attn_masks},
                    ]

                    logger.log_spec(report_mels_keys, coarse_olens, gs, report_name_for_outs, str(valid_num))
                    logger.log_attn_ws(report_attn_ws_keys, ilens, coarse_olens, gs, report_name_for_outs, str(valid_num))

            return loss, report_keys_loss

        else:
            mels_pred = self.ssrn(coarse_mels)
            loss, report_keys_loss = self.DCTTSLoss(mels_pred, mels.detach())

            if logger is not None:
                if not valid:
                    if gs % int(self.loss_snapshot_step) == 0:
                        logger.log_loss(report_keys_loss, gs)

                else:
                    olens = batch['olens'].detach().to(device)
                    report_mels_keys = [{"true_mel": mels.transpose(1, 2)},
                                        {"mel_pred": mels_pred[1].transpose(1, 2)}]

                    logger.log_spec(report_mels_keys, olens, gs, report_name_for_outs, str(valid_num))

            return loss, report_keys_loss

    def _inference(self, text=None, pred_mels=None, batch=None, device=None):
        if self.train_module == 't2m':
            if text is None:
                texts = batch['xs'].detach().to(device)
                mels = torch.FloatTensor(np.zeros((len(batch['xs']), 80, self.max_target_len))).to(device)
            else:
                # texts = text.unsqueeze(0)
                texts = torch.LongTensor(1, len(text))
                texts[0] = text
                mels = torch.FloatTensor(np.zeros((1, 80, 150))).to(device)

            prev_attn = None
            # for t in range(self.max_target_len - 1):
            for t in range(149):
                ks, vs = self.text_encoder(texts)
                zs, attn_ws = self.audio_encoder(ks, vs, mels, None if t == 0 else t - 1, prev_attn)
                _, new_mel = self.audio_decoder(zs)
                mels[:, :, t + 1].data.copy_(new_mel[:, :, t].data)
                prev_attn = attn_ws
            pred_mels = mels
        else:
            pred_mels = self.ssrn(pred_mels)
            attn_ws = None
        return pred_mels, attn_ws

def optimizer(conf, dctts):

    conf_opt = conf['optimizer']
    optimizer = torch.optim.Adam(
        dctts.parameters(),
        lr=float(conf_opt['adam_alpha']),
        betas=(float(conf_opt['adam_beta1']), float(conf_opt['adam_beta2'])),
        eps=float(conf_opt['adam_eps'])
    )

    optimizers = {"optimizer": optimizer}
    return optimizers


class SuperRes(nn.Module):
    def __init__(self, fdim, ddim, dropout_rate):
        torch.nn.Module.__init__(self)

        self.convs = torch.nn.ModuleList()

        self.convs += [MaskedConv1d(in_channels=fdim,
                                    out_channels=ddim,
                                    kernel_size=1,
                                    padding="same"),
                       nn.Dropout(dropout_rate)]

        for _ in range(1):
            for j in range(2):
                self.convs += [HighwayConv1d(ddim,
                                             kernel_size=3,
                                             dilation=3 ** j,
                                             padding="same"),
                               nn.Dropout(dropout_rate)]

        for _ in range(2):
            self.convs += [Deconv1d(in_channels=ddim,
                                    out_channels=ddim,
                                    kernel_size=2,
                                    padding="same"),
                           nn.Dropout(dropout_rate)]

            for j in range(2):
                self.convs += [HighwayConv1d(ddim,
                                             kernel_size=3,
                                             dilation=3 ** j,
                                             padding="same"),
                               nn.Dropout(dropout_rate)]

        self.convs += [MaskedConv1d(in_channels=ddim,
                                    out_channels=fdim,
                                    kernel_size=1,
                                    padding="same"),
                       nn.Dropout(dropout_rate)]

        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        for f in self.convs:
            xs = f(xs)
        _xs = self.sigmoid(xs)
        return xs, _xs


class TextEncoder(nn.Module):
    def __init__(self, idim, edim, ddim, dropout_rate):
        torch.nn.Module.__init__(self)

        self.embed = nn.Embedding(idim, edim)
        self.convs = torch.nn.ModuleList()
        self.convs += [MaskedConv1d(in_channels=edim,
                                 out_channels=ddim * 2,
                                 kernel_size=1,
                                 padding="same"),
                       nn.ReLU(),
                       nn.Dropout(dropout_rate),
                       MaskedConv1d(in_channels=ddim * 2,
                                 out_channels=ddim * 2,
                                 kernel_size=1,
                                 padding="same"),
                       nn.Dropout(dropout_rate)]

        for _ in range(2):
            for j in range(4):
                self.convs += [HighwayConv1d(in_channels=ddim * 2,
                                         kernel_size=3,
                                         dilation=3 ** j,
                                             padding="same"),
                               nn.Dropout(dropout_rate)]
        for j in [1, 0]:
            for k in range(2):
                self.convs += [HighwayConv1d(in_channels=ddim * 2,
                                             kernel_size=3 ** j,
                                             dilation=1,
                                             padding="same")]
                if not (j == 0 and k == 1):
                    nn.Dropout(dropout_rate)


    def forward(self, xs):
        xs = self.embed(xs).transpose(1, 2)
        for f in self.convs:
            xs = f(xs)
        ks, vs = torch.chunk(xs, 2, 1)
        return ks, vs


class AudioEncoder(nn.Module):
    def __init__(self, fdim, ddim, dropout_rate, is_training=True):
        torch.nn.Module.__init__(self)

        self.is_training = is_training
        self.ddim = ddim
        self.convs = torch.nn.ModuleList()

        self.convs += [MaskedConv1d(in_channels=fdim,
                                    out_channels=ddim,
                                    kernel_size=1,
                                    padding="causal"),
                       nn.ReLU(),
                       nn.Dropout(dropout_rate),
                       MaskedConv1d(in_channels=ddim,
                                    out_channels=ddim,
                                    kernel_size=1,
                                    padding="causal"),
                       nn.ReLU(),
                       nn.Dropout(dropout_rate),
                       MaskedConv1d(in_channels=ddim,
                                    out_channels=ddim,
                                    kernel_size=1,
                                    padding="causal"),
                       nn.Dropout(dropout_rate),
                       ]

        for _ in range(2):
            for j in range(4):
                self.convs += [HighwayConv1d(ddim,
                                             kernel_size=3,
                                             dilation=3 ** j,
                                             padding="causal"),
                               nn.Dropout(dropout_rate)]
        for k in range(2):
            self.convs += [HighwayConv1d(ddim,
                                         kernel_size=3,
                                         dilation=3,
                                         padding="causal")]

            if k == 0:
                self.convs += [nn.Dropout(dropout_rate)]

    def forward(self, ks, vs, xs, prev_time=None, prev_atten=None):
        for f in self.convs:
            xs = f(xs)
        attn_ws = nn.functional.softmax(torch.bmm(ks.transpose(1, 2), xs) / np.sqrt(self.ddim), 1)
        if not (self.is_training or prev_time is None or prev_atten is None):
            # forcibly incremental attention, at inference phase
            attn_ws[:, :, :prev_time+1].data.copy_(prev_atten[:, :, :prev_time+1].data)
            for i in range(int(attn_ws.size(0))):
                if prev_time == 0 :
                    nt0 = 0
                    nt1 = 1
                else:
                    nt0 = torch.argmax(attn_ws[i, :, prev_time])
                    nt1 = torch.argmax(attn_ws[i, :, prev_time + 1])
                # print(prev_time, nt0.cpu().data, nt1.cpu().data)
                if nt1 < nt0 - 1 or nt1 > nt0 + 3:
                    nt0 = nt0 if nt0 + 1 < attn_ws.size(1) else nt0 - 1
                    attn_ws[i, :, prev_time + 1].zero_()
                    attn_ws[i, nt0 + 1, prev_time + 1] = 1
        xs = torch.cat((torch.bmm(vs, attn_ws), xs), 1)
        return xs, attn_ws


class AudioDecoder(nn.Module):
    def __init__(self, fdim, ddim, dropout_rate):
        torch.nn.Module.__init__(self)

        self.convs = torch.nn.ModuleList()
        self.convs += [MaskedConv1d(ddim * 2, ddim, 1, padding="causal"),
                       nn.Dropout(dropout_rate)]

        for _ in range(1):
            for j in range(4):
                self.convs += [HighwayConv1d(ddim,
                                             kernel_size=3,
                                             dilation=3 ** j,
                                             padding="causal"),
                               nn.Dropout(dropout_rate)]

        for _ in range(2):
            self.convs += [HighwayConv1d(ddim,
                                         kernel_size=3,
                                         dilation=1,
                                         padding="causal"),
                           nn.Dropout(dropout_rate)]

        for _ in range(3):
            self.convs += [MaskedConv1d(in_channels=ddim,
                                        out_channels=ddim,
                                        kernel_size=1,
                                        dilation=1,
                                        padding="causal"),
                           nn.ReLU(),
                           nn.Dropout(dropout_rate)]

        self.convs += [MaskedConv1d(in_channels=ddim,
                                    out_channels=fdim,
                                    kernel_size=1,
                                    dilation=1,
                                    padding="causal")]

        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        for f in self.convs:
            xs = f(xs)
        _xs = self.sigmoid(xs)
        return xs, _xs