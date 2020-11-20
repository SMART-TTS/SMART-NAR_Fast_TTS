import numpy as np
from torch.optim import lr_scheduler

from model.tts.dctts.module.conv import *


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


class ModelLoss(nn.Module):
    def __init__(self, conf):
        torch.nn.Module.__init__(self)

        # reconstruction loss
        self.criterion_recon = nn.L1Loss()
        self.criterion_attn = nn.L1Loss()
        self.criterion_feat_match = nn.L1Loss()

        self.num_D = conf['model']['d']['num_D']

    def forward(self, step, ys_pred=None, ys=None, rs_pred=None, rs=None,
                attn_ws=None, attn_guide=None, attn_mask=None, dynamic_guide=None, d_out_fake=None, d_out_real=None):

        if step == 'g':
            attn_mask = torch.ones_like(attn_guide)
            loss_recon_coarse_mels_teacher = self.criterion_recon(ys_pred[0][1], ys[0])
            loss_recon_coarse_mels_student = self.criterion_recon(ys_pred[1][1], ys[0])
            loss_recon_mels_teacher = self.criterion_recon(ys_pred[2][1][:, :, :ys[1].size(2)], ys[1])
            loss_recon_mels_student = self.criterion_recon(ys_pred[3][1][:, :, :ys[1].size(2)], ys[1])

            loss_attn_mel_enc_layer1 = self.criterion_attn(attn_guide * attn_ws[0][0] * attn_mask, torch.zeros_like(attn_guide)) * dynamic_guide
            loss_attn_mel_enc_layer2 = self.criterion_attn(attn_guide * attn_ws[0][1] * attn_mask, torch.zeros_like(attn_guide)) * dynamic_guide
            loss_attn_mel_dec = self.criterion_attn(attn_guide * attn_ws[1] * attn_mask, torch.zeros_like(attn_guide)) * dynamic_guide
            loss_attn_mel_generator = self.criterion_attn(attn_guide * attn_ws[2] * attn_mask, torch.zeros_like(attn_guide)) * dynamic_guide

            g_loss = 0.0
            for i, scale in enumerate(d_out_fake):
                g_loss += -scale[-1].mean()

            feat_mat_loss = 0.0
            wt = 10.0 / 4.0
            for i in range(self.num_D):
                for j in range(len(d_out_fake[i]) - 1):
                    feat_mat_loss += wt * self.criterion_feat_match(d_out_fake[i][j], d_out_real[i][j].detach())

            loss = loss_recon_coarse_mels_student + loss_recon_coarse_mels_teacher + \
                   loss_recon_mels_teacher + loss_recon_mels_student + \
                   loss_attn_mel_enc_layer1 + loss_attn_mel_enc_layer2 + \
                   loss_attn_mel_dec + loss_attn_mel_generator + \
                   g_loss + feat_mat_loss

            report_loss_keys = [
                {"loss_recon_coarse_mels_student": loss_recon_coarse_mels_student.item()},
                {"loss_recon_coarse_mels_teacher": loss_recon_coarse_mels_teacher.item()},
                {"loss_recon_mels_teacher": loss_recon_mels_teacher.item()},
                {"loss_recon_mels_student": loss_recon_mels_student.item()},
                {"loss_attn_mel_enc_layer1": loss_attn_mel_enc_layer1.item()},
                {"loss_attn_mel_enc_layer2": loss_attn_mel_enc_layer2.item()},
                {"loss_attn_mel_dec": loss_attn_mel_dec.item()},
                {"loss_attn_mel_generator": loss_attn_mel_generator.item()},
                {"g_loss": g_loss.item()},
                {"loss_feat_match": feat_mat_loss.item()},
                {"loss": loss.item()}]

        else:
            d_loss = 0
            for i, scale in enumerate(d_out_fake):
                d_loss += nn.functional.relu(1 + scale[-1]).mean()

            for i, scale in enumerate(d_out_real):
                d_loss += nn.functional.relu(1 - scale[-1]).mean()

            loss = d_loss
            report_loss_keys = [
                {"d_loss": d_loss},
            ]

        return loss, report_loss_keys


class Model(nn.Module):
    def __init__(self, conf, is_training=True):
        super(Model, self).__init__()

        self.is_training = is_training
        if conf['train']['attention_masking'] is not None:
            self.is_attention_masking = True
        self.DCGANTTSLoss = ModelLoss(conf)
        self.loss_snapshot_step = conf['train']['loss_snapshot_step']

        self.main_device = conf['train']['device'][0]
        self.conf_model = conf['model']
        self.conf_data = conf['data']
        self.idim = self.conf_model['idim']
        self.fdim = self.conf_model['fdim']
        self.edim = self.conf_model['edim']
        self.ddim = self.conf_model['ddim']
        self.dropout_rate = self.conf_model['dropout_rate']
        self.drop_mel = self.conf_model['drop_mel']
        self.length_ratio_max = self.conf_data['length_ratio_max']
        self.length_ratio_min = self.conf_data['length_ratio_min']
        self.disc_in_level = self.conf_model['discriminator_input_level']

        self.text_encoder = TextEncoder(self.idim, self.edim, self.ddim, self.dropout_rate)
        self.mel_decoder = AudioDecoder(self.fdim, self.ddim, self.dropout_rate)
        self.postnet = PostNet(self.fdim, self.ddim, self.dropout_rate)
        self.generator = Generator(self.ddim, self.dropout_rate)
        if is_training:
            self.mel_encoder = AudioEncoder(self.fdim, self.ddim, self.dropout_rate)
            self.discriminator = Discriminator(**self.conf_model['d'])

#    def forward(self, step, batch=None, valid=False, d_in_fake=None, d_in_real=None, dynamic_guide=None, logger=None, gs=None, valid_num=None, report_name_for_outs=None):
    def forward(self, step=None, batch=None, d_in_real=None, d_in_fake=None, logger=None, gs=None, epoch=None,
                dynamic_guide=None, voc=None,
                valid=False, valid_num=None, report_name_for_outs=None,
                texts=None, target_len=None):
        if self.is_training:
            return self._forward(step=step, batch=batch, d_in_real=d_in_real, d_in_fake=d_in_fake,
                                 valid=valid, dynamic_guide=dynamic_guide, logger=logger, gs=gs, epoch=epoch,
                                 valid_num=valid_num, report_name_for_outs=report_name_for_outs,
                                 voc=voc)
        else:
            return self._inference(batch=batch)

    def _forward(self, step, batch, d_in_real=None, d_in_fake=None,
                 valid=False, dynamic_guide=None, logger=None, gs=None, epoch=None,
                 valid_num=None, report_name_for_outs=None, voc=None):
        report_loss_keys = []
        coarse_olens = batch['coarse_olens'].detach().cuda()
        coarse_olens_max = max(coarse_olens)
        olens = batch['olens'].detach().cuda()
        olens_max = max(olens)
        ilens = batch['ilens'].detach().cuda()
        ilens_max = max(ilens)
        if step == 'g':
            texts = batch['text'].detach().cuda()
            mels = batch['mel'].detach().cuda()
            coarse_mels = batch['coarse_mel'].detach().cuda()
            attn_guides = batch['attn_guide'].detach().cuda()
            coarse_mels_in = nn.functional.dropout(coarse_mels, p=self.drop_mel, training=True)
            zs = [np.random.normal(0, 1, (olen, 1)) for olen in coarse_olens]
            zs = pad_list([torch.from_numpy(z).float() for z in zs], 0).cuda()
            if self.is_attention_masking and epoch < 100:
                attn_mask_for_attn_masking = batch['attn_mask2'].detach().cuda()
                attn_mask_for_attn_masking = attn_mask_for_attn_masking[:, :ilens_max, :coarse_olens_max]
            else:
                attn_mask_for_attn_masking = None

            ks, vs = self.text_encoder(texts)
            ls_from_mel_enc, attn_ws_mel_enc = self.mel_encoder(ks, vs, coarse_mels_in,
                                                                attn_mask_for_attn_masking)
            coarse_mels_pred_from_mel_enc, attn_ws_mel_dec = self.mel_decoder(ks, vs, ls_from_mel_enc,
                                                                              attn_mask_for_attn_masking)
            mels_pred_from_mel_enc = self.postnet(coarse_mels_pred_from_mel_enc[1])

            ls_from_generator, attn_ws_generator = self.generator(ks, vs, zs,
                                                                  attn_mask_for_attn_masking)
            coarse_mels_pred_from_generator, attn_ws_generator = self.mel_decoder(ks, vs, ls_from_generator,
                                                                                  attn_mask_for_attn_masking)
            mels_pred_from_generator = self.postnet(coarse_mels_pred_from_generator[1])

            ys_pred = (coarse_mels_pred_from_mel_enc, coarse_mels_pred_from_generator, mels_pred_from_mel_enc,
                       mels_pred_from_generator)
            ys = (coarse_mels, mels)
            attn_ws = (attn_ws_mel_enc, attn_ws_mel_dec, attn_ws_generator)

            if self.disc_in_level == 'mel':
                d_in_real = coarse_mels_pred_from_mel_enc[1]
                d_in_fake = coarse_mels_pred_from_generator[1]
            else:
                d_in_real = ls_from_mel_enc
                d_in_fake = ls_from_generator

            d_out_fake = self.discriminator(d_in_fake)
            d_out_real = self.discriminator(d_in_real.detach())
            loss, _report_loss_keys = self.DCGANTTSLoss(step=step, ys_pred=ys_pred, ys=ys,
                                                        attn_ws=attn_ws, attn_guide=attn_guides,
                                                        dynamic_guide=dynamic_guide,
                                                        d_out_fake=d_out_fake, d_out_real=d_out_real)
            report_loss_keys += _report_loss_keys


        if step == 'd':
            d_out_fake = self.discriminator(d_in_fake.detach())
            d_out_real = self.discriminator(d_in_real.detach())
            loss, _report_loss_keys = self.DCGANTTSLoss(step=step, d_out_fake=d_out_fake, d_out_real=d_out_real)
            report_loss_keys += _report_loss_keys

        if logger is not None:
            if not valid:
                if gs % int(self.loss_snapshot_step) == 0:
                    logger.log_loss(report_loss_keys, gs)

            else:
                report_mels_keys = []
                report_attn_ws_keys = []
                report_mels_keys += [{"coarse_mels_pred_from_mel_enc": coarse_mels_pred_from_mel_enc[1].transpose(1, 2)},
                                     {"coarse_mels_pred_from_generator": coarse_mels_pred_from_generator[1].transpose(1, 2)},
                                     {"coarse_mels_true": coarse_mels.transpose(1, 2)},
                                     {"ls_from_mel_enc": ls_from_mel_enc.transpose(1, 2)},
                                     {"ls_from_generator": ls_from_generator.transpose(1, 2)},
                                     ]
                report_attn_ws_keys += [{"attn_ws_mel_enc_layer_1": attn_ws_mel_enc[0]},
                                        {"attn_ws_mel_enc_layer_2": attn_ws_mel_enc[1]},
                                        {"attn_ws_mel_dec": attn_ws_mel_dec},
                                        {"attn_ws_mel_generator": attn_ws_generator},
                                        {"attn_guide": attn_guides},
                                        ]

                logger.log_spec(report_mels_keys, coarse_olens, gs, report_name_for_outs, str(valid_num))
                logger.log_attn_ws(report_attn_ws_keys, ilens, coarse_olens, gs, report_name_for_outs, str(valid_num))

        return loss, d_in_real, d_in_fake


def optimizer(conf, dcgantts):
    conf_optimizer = conf['optimizer']
    conf_scheduler = conf['scheduler']

    for name, param in dcgantts.named_parameters():
        if 'discriminator' in name:
            param.requires_grad = False
    optimizer_g = torch.optim.Adam(
        filter(lambda p: p.requires_grad, dcgantts.parameters()),
        lr=float(conf_optimizer['adam_alpha']),
        betas=(float(conf_optimizer['adam_beta1']), float(conf_optimizer['adam_beta2'])))
    scheduler_g = lr_scheduler.ExponentialLR(optimizer_g, gamma=conf_scheduler['gamma'])

    for name, param in dcgantts.named_parameters():
        param.requires_grad = False
        if 'discriminator' in name:
            param.requires_grad = True
    optimizer_d = torch.optim.Adam(
        filter(lambda p: p.requires_grad, dcgantts.parameters()),
        lr=float(conf_optimizer['adam_alpha']),
        betas=(float(conf_optimizer['adam_beta1']), float(conf_optimizer['adam_beta2'])))
    scheduler_d = lr_scheduler.ExponentialLR(optimizer_d, gamma=conf_scheduler['gamma'])

    for name, param in dcgantts.named_parameters():
        param.requires_grad = True

    optimizers = {"optimizer_g": optimizer_g,
                  "optimizer_d": optimizer_d}

    schedulers = {"scheduler_g": scheduler_g,
                  "scheduler_d": scheduler_d}

    return optimizers, schedulers


class DurationPredictor(nn.Module):
    def __init__(self, ddim, dropout_rate):
        torch.nn.Module.__init__(self)

        self.convs = torch.nn.ModuleList()
        self.convs += [MaskedConv1d(in_channels=ddim,
                                    out_channels=ddim,
                                    kernel_size=1,
                                    padding="same"),
                       nn.ReLU(),
                       nn.Dropout(dropout_rate),
                       MaskedConv1d(in_channels=ddim,
                                    out_channels=ddim,
                                    kernel_size=1,
                                    padding="same"),
                       nn.Dropout(dropout_rate)]

        self.linear = nn.Linear(ddim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for f in self.convs:
            x = f(x)
        x = self.linear(x.transpose(1, 2))
        x = self.sigmoid(x)
        return x.transpose(1, 2)


class PostNet(nn.Module):
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
        xs_ = self.sigmoid(xs)
        return xs, xs_


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
                    self.convs += [nn.Dropout(dropout_rate)]

    def forward(self, xs):
        xs = self.embed(xs).transpose(1, 2)
        for f in self.convs:
            xs = f(xs)
        ks, vs = torch.chunk(xs, 2, 1)
        return ks, vs


class AudioEncoder(nn.Module):
    def __init__(self, fdim, ddim, dropout_rate):
        torch.nn.Module.__init__(self)

        self.ddim = ddim
        self.convs1 = torch.nn.ModuleList()
        self.convs2 = torch.nn.ModuleList()
        self.convs3 = torch.nn.ModuleList()

        self.convs1 += [MaskedConv1d(in_channels=fdim,
                                    out_channels=ddim,
                                    kernel_size=1,
                                    padding="same"),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        MaskedConv1d(in_channels=ddim,
                                     out_channels=ddim,
                                     kernel_size=1,
                                     padding="same"),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        MaskedConv1d(in_channels=ddim,
                                     out_channels=ddim,
                                     kernel_size=1,
                                     padding="same"),
                        nn.Dropout(dropout_rate),
                        ]

        self.convs2 += [MaskedConv1d(in_channels=ddim * 2,
                                   out_channels=ddim,
                                   kernel_size=1,
                                   padding="same"),
                        nn.Dropout(dropout_rate),
                        ]
        for k in range(2):
            for j in range(4):
                self.convs2 += [HighwayConv1d(ddim,
                                              kernel_size=3,
                                              dilation=3 ** j,
                                              padding="same"),
                                nn.Dropout(dropout_rate)]

        self.convs3 += [MaskedConv1d(in_channels=ddim * 2,
                                     out_channels=ddim,
                                     kernel_size=1,
                                     padding="same"),
                        nn.Dropout(dropout_rate),
                        ]
        for k in range(2):
            self.convs3 += [HighwayConv1d(ddim,
                                          kernel_size=3,
                                          dilation=3,
                                          padding="same"),
                            nn.Dropout(dropout_rate)]

        self.convs3 += [MaskedConv1d(in_channels=ddim,
                                   out_channels=64,
                                   kernel_size=1,
                                   dilation=1,
                                   padding="same")]

    def forward(self, ks, vs, xs, attn_mask=None):
        attn_ws = []
        for f in self.convs1:
            xs = f(xs)

        if attn_mask is None:
            a = nn.functional.softmax(torch.bmm(ks.transpose(1, 2), xs) / np.sqrt(self.ddim), 1)
        else:
            _a = (torch.bmm(ks.transpose(1, 2), xs) / np.sqrt(self.ddim)) * attn_mask
            a = nn.functional.softmax(_a, 1)

        xs = torch.cat((torch.bmm(vs, a), xs), 1)
        attn_ws.append(a)

        for f in self.convs2:
            xs = f(xs)

        if attn_mask is None:
            a = nn.functional.softmax(torch.bmm(ks.transpose(1, 2), xs) / np.sqrt(self.ddim), 1)
        else:
            _a = (torch.bmm(ks.transpose(1, 2), xs) / np.sqrt(self.ddim)) * attn_mask
            a = nn.functional.softmax(_a, 1)

        xs = torch.cat((torch.bmm(vs, a), xs), 1)
        attn_ws.append(a)

        for f in self.convs3:
            xs = f(xs)
        return xs, attn_ws


class AudioDecoder(nn.Module):
    def __init__(self, fdim, ddim, dropout_rate):
        torch.nn.Module.__init__(self)

        self.ddim = ddim
        self.convs1 = torch.nn.ModuleList()
        self.convs1 += [MaskedConv1d(64, ddim, 1, padding="same"),
                        nn.Dropout(dropout_rate)]

        for i in range(1):
            for j in range(4):
                self.convs1 += [HighwayConv1d(ddim,
                                              kernel_size=3,
                                              dilation=3 ** j,
                                              padding="same")]

                if not (i == 1 and j == 3):
                    self.convs1 += [nn.Dropout(dropout_rate)]

        self.convs2 = torch.nn.ModuleList()
        self.convs2 += [MaskedConv1d(in_channels=ddim * 2,
                                     out_channels=ddim,
                                     kernel_size=1,
                                     padding="same"),
                        nn.Dropout(dropout_rate),
                        ]

        for _ in range(2):
            self.convs2 += [HighwayConv1d(ddim,
                                          kernel_size=3,
                                          dilation=1,
                                          padding="same"),
                            nn.Dropout(dropout_rate)]

        for _ in range(3):
            self.convs2 += [MaskedConv1d(in_channels=ddim,
                                         out_channels=ddim,
                                         kernel_size=1,
                                         dilation=1,
                                         padding="same"),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate)]

        self.convs2 += [MaskedConv1d(in_channels=ddim,
                                     out_channels=fdim,
                                     kernel_size=1,
                                     dilation=1,
                                     padding="same")]

        self.sigmoid = nn.Sigmoid()

    def forward(self, ks, vs, xs, attn_mask=None):
        for f in self.convs1:
            xs = f(xs)

        if attn_mask is None:
            attn_ws = nn.functional.softmax(torch.bmm(ks.transpose(1, 2), xs) / np.sqrt(self.ddim), 1)
        else:
            _attn_ws = (torch.bmm(ks.transpose(1, 2), xs) / np.sqrt(self.ddim)) * attn_mask
            attn_ws = nn.functional.softmax(_attn_ws, 1)

        xs = torch.cat((torch.bmm(vs, attn_ws), xs), 1)

        for f in self.convs2:
            xs = f(xs)

        xs_ = self.sigmoid(xs)
        return (xs, xs_), attn_ws


class Generator(nn.Module):
    def __init__(self, ddim, dropout_rate, is_training=True):
        torch.nn.Module.__init__(self)

        self.is_training = is_training
        self.ddim = ddim
        self.convs1 = torch.nn.ModuleList()
        self.convs1 += [MaskedConv1d(in_channels=1,
                                     out_channels=ddim,
                                     kernel_size=1,
                                     padding="same"),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        MaskedConv1d(in_channels=ddim,
                                     out_channels=ddim,
                                     kernel_size=1,
                                     padding="same"),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        MaskedConv1d(in_channels=ddim,
                                     out_channels=ddim,
                                     kernel_size=1,
                                     padding="same"),
                        nn.Dropout(dropout_rate),
                        ]

        for _ in range(2):
            for j in range(4):
                self.convs1 += [HighwayConv1d(ddim,
                                            kernel_size=3,
                                            dilation=3 ** j,
                                            padding="same"),
                              nn.Dropout(dropout_rate)]

        for k in range(2):
            self.convs1 += [HighwayConv1d(ddim,
                                          kernel_size=3,
                                          dilation=3,
                                          padding="same")]

            if k == 0:
                self.convs1 += [nn.Dropout(dropout_rate)]


        self.convs2 = torch.nn.ModuleList()
        self.convs2 += [MaskedConv1d(in_channels=ddim * 2,
                                     out_channels=ddim,
                                     kernel_size=1,
                                     padding="same"),
                        nn.Dropout(dropout_rate),
                        ]

        for _ in range(1):
            for j in range(4):
                self.convs2 += [HighwayConv1d(ddim,
                                              kernel_size=3,
                                              dilation=3 ** j,
                                              padding="same"),
                              nn.Dropout(dropout_rate)]

        for _ in range(2):
            self.convs2 += [HighwayConv1d(ddim,
                                          kernel_size=3,
                                          dilation=1,
                                          padding="same"),
                          nn.Dropout(dropout_rate)]

        for _ in range(3):
            self.convs2 += [MaskedConv1d(in_channels=ddim,
                                         out_channels=ddim,
                                         kernel_size=1,
                                         dilation=1,
                                         padding="causal"),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate)]

        self.convs2 += [MaskedConv1d(in_channels=ddim,
                                     out_channels=64,
                                     kernel_size=1,
                                     dilation=1,
                                     padding="causal")]

    def forward(self, ks, vs, xs, attn_mask=None):
        xs = xs.transpose(1, 2)
        for f in self.convs1:
            xs = f(xs)
        if attn_mask is None:
            attn_ws = nn.functional.softmax(torch.bmm(ks.transpose(1, 2), xs) / np.sqrt(self.ddim), 1)
        else:
            _attn_ws = (torch.bmm(ks.transpose(1, 2), xs) / np.sqrt(self.ddim)) * attn_mask
            attn_ws = nn.functional.softmax(_attn_ws, 1)
        xs = torch.cat((torch.bmm(vs, attn_ws), xs), 1)
        for f in self.convs2:
            xs = f(xs)
        return xs, attn_ws


class NLayerDiscriminator(nn.Module):
    def __init__(self, idim, n_layers):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.Conv1d(idim, 128, kernel_size=1, stride=1, dilation=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )

        model["layer_1"] = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=1, dilation=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        model["layer_2"] = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, stride=1, dilation=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )

        model["layer_3"] = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, stride=1, dilation=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )

        model["layer_4"] = nn.Sequential(
            nn.Conv1d(1024, 1, kernel_size=1, stride=1, dilation=1),
            nn.Sigmoid()
        )
        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, num_D, idim, n_layers_D):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                idim, n_layers_D
            )
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            if len(self.model) > 1:
                x = self.downsample(x)
        return results
