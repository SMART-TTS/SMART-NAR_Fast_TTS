import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import TxtEncoder, MelEncoder, MelDecoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2Align(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Align, self).__init__()
        self.model_config = model_config

        self.txt_encoder = TxtEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.mel_encoder = MelEncoder(model_config)
        self.mel_decoder = MelDecoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        p_control=1.0,
        e_control=1.0,
    ):
        is_training = False if mel_lens is None else True

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        src_output = self.txt_encoder(texts, src_masks)

        if is_training:
            tgt_output, tgt_alignment = self.mel_encoder(src_output, mels, src_masks, mel_masks)
            d_targets = torch.stack([self._calculate_duration(attn, src_len, mel_len, max_src_len)
                                     for attn, src_len, mel_len in zip(tgt_alignment[-1].detach(), src_lens, mel_lens)])
        else:
            d_targets, tgt_alignment = None, None

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            src_output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
        )

        output, mel_masks = self.mel_decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            tgt_alignment,
            d_targets,
        )
