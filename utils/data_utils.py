import os
import random
import numpy as np
import torch
import torch.utils.data

import librosa
import torch.nn.functional as F

from utils import load_wav_to_torch, load_filepaths_and_text, layers, TimeVaryingMetaTemplate
from utils.text import text_to_sequence
from utils.spectrogram import logmelspectrogram
from utils.layers import TacotronSTFT

class _DataLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths, conf, valid=False):
        data_conf = conf['data']
        self.is_norm = data_conf['is_norm']
        self.is_valid = valid

        if 'mel' in data_conf['batch']:
            self.use_mel = True
        else:
            self.use_mel = False
        self.data_name = data_conf['data_name']

        self.sampling_rate = data_conf['sampling_rate'] if data_conf['sampling_rate'] else None
        self.n_fft = data_conf['n_fft'] if data_conf['n_fft'] else None
        self.hop_length = data_conf['hop_length'] if data_conf['hop_length'] else None
        self.win_length = data_conf['win_length'] if data_conf['win_length'] else None
        self.n_mel = data_conf['n_mel'] if data_conf['n_mel'] else None
        self.audio_refdB = data_conf['audio_refdB'] if data_conf['audio_refdB'] else None
        self.audio_maxdB = data_conf['audio_maxdB'] if data_conf['audio_maxdB'] else None
        self.reduction_factor = data_conf['reduction_factor'] if data_conf['reduction_factor'] else None
        self.segment_length = data_conf['segment_length'] if data_conf['segment_length'] else None
        self.text_cleaners = data_conf['text_cleaners'] if data_conf['text_cleaners'] else None

        self.use_audio = True if 'audio' in data_conf['batch'] else False
        self.use_audio_seg = True if 'audio_seg' in data_conf['batch'] else False
        self.use_mel_seg = True if 'mel_seg' in data_conf['batch'] else False
        self.use_coarse_mel = True if self.reduction_factor is not None and self.reduction_factor > 1 else False
        self.use_mel = True if 'mel' in data_conf['batch'] else False
        self.use_text = True if 'text' in data_conf['batch'] else False
        self.use_attn_guide = True if 'attn_guide' in data_conf['batch'] else False
        self.use_attn_mask = True if 'attn_mask' in data_conf['batch'] else False
        self.use_tvmt = True if 'tvmt' in data_conf['batch'] else False
        self.use_attn_mask2 = True if 'attn_mask2' in data_conf['batch'] else False

        self.load_mel_from_disk = conf['load_mel_from_disk']
        self.audiopaths = load_filepaths_and_text(audiopaths)

        random.seed(1234)
        random.shuffle(self.audiopaths)

    def get_segment_wav(self, audiopath):
        audio, rate = self.get_wav(audiopath)
        # Take segment
        if audio.shape[0] >= self.segment_length:
            max_audio_start = audio.shape[0] - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio_seg = audio[audio_start : audio_start + self.segment_length]
        else:
            audio_seg = F.pad(
                audio, (0, self.segment_length - audio.shape[0]), "constant"
            ).data

        return audio_seg, rate

    def get_wav(self, audiopath):
        audio, rate = librosa.core.load(audiopath, sr=self.sampling_rate)
        audio = 0.95 * librosa.util.normalize(audio)

        # audio = (audio.numpy() * 32768).astype("int16")
        # scipy.io.wavfile.write(file_path, sampling_rate, audio)

        return audio, rate

    def get_mel(self, audiopath=None, audio=None, r=1):
        if self.load_mel_from_disk:
            if self.is_norm:
                if r == 1:
                    if 'chj' in self.data_name:
                        mel_path = audiopath.replace('db/{}'.format(self.data_name), 'feats/{}/mels_norm'.format(self.data_name))
                        mel_path = mel_path.replace('wav', 'npy')
                    elif self.data_name == 'lj':
                        mel_path = os.path.join('feats', 'lj', 'mels_norm', '{}.npy'.format(audiopath))
                    mel = np.load(mel_path)

                else:
                    if 'chj' in self.data_name:
                        mel_path = audiopath.replace('db/{}'.format(self.data_name), 'feats/{}/coarse_mels_norm'.format(self.data_name))
                        mel_path = mel_path.replace('wav', 'npy')
                    elif self.data_name == 'lj':
                        mel_path = os.path.join('feats', 'lj', 'coarse_mels_norm', '{}.npy'.format(audiopath))
                    mel = np.load(mel_path)

        else:
            if self.is_norm:
                if audio is None:
                    audio, _ = self.get_wav(audiopath)
                spec = librosa.stft(y=audio,
                                    n_fft=int(self.n_fft),
                                    hop_length=int(self.hop_length),
                                    win_length=int(self.win_length))
                spec = np.absolute(spec)
                mel_filters = librosa.filters.mel(self.sampling_rate, self.n_fft, self.n_mel)
                mel = np.dot(mel_filters, spec)
                # to dB
                mel[mel < 1e-10] = 1e-10
                mel = 20 * np.log10(mel)
                # normalize
                mel = np.clip((mel - self.audio_refdB + self.audio_maxdB) / self.audio_maxdB, 1e-8, 1)

                if r > 1:
                    pad = r - (mel.shape[1] % r) if mel.shape[1] % r != 0 else 0
                    mel = np.pad(mel, [[0, 0], [0, pad]], mode="constant")

        return mel

    def get_text(self, text):
        # text_norm = torch.IntTensor(text_to_sequence(text.rstrip(), self.text_cleaners))
        text_norm = torch.IntTensor(text_to_sequence(text.rstrip()))
        return text_norm

    def get_attn_guides_and_masks(self, audiopath):
        if 'chj' in self.data_name:
            guide_path = audiopath.replace('db/{}'.format(self.data_name), 'feats/{}/guide'.format(self.data_name))
            guide_path = guide_path.replace('wav', 'npy')
            mask_path = audiopath.replace('db/{}'.format(self.data_name), 'feats/{}/mask'.format(self.data_name))
            mask_path = mask_path.replace('wav', 'npy')
            mask2_path = audiopath.replace('db/{}'.format(self.data_name), 'feats/{}/mask_for_attn_masking'.format(self.data_name))
            mask2_path = mask2_path.replace('wav', 'npy')
        elif self.data_name == 'lj':
            guide_path = os.path.join('feats', 'lj', 'guide', '{}.npy'.format(audiopath))
            mask_path = os.path.join('feats', 'lj', 'mask', '{}.npy'.format(audiopath))
        attn_guide = np.load(guide_path)
        attn_mask = np.load(mask_path)
        attn_mask2 = np.load(mask2_path)

        return (torch.from_numpy(attn_guide), torch.from_numpy(attn_mask), torch.from_numpy(attn_mask2))

    def get_time_varying_meta_template(self, audiopath):
        if 'chj' in self.data_name:
            tvmt_path = audiopath.replace('db/{}'.format(self.data_name), 'feats/{}/tvmt'.format(self.data_name))
            tvmt_path = tvmt_path.replace('wav', 'npy')
        elif self.data_name == 'lj':
            tvmt_path = os.path.join('feats', 'lj', 'tvmt', '{}.npy'.format(audiopath))
        tvmt = np.load(tvmt_path)
        return torch.from_numpy(tvmt)

    def get_batch(self, audiopath_and_text):
        audiopath = audiopath_and_text[0]
        text = audiopath_and_text[1]
        if self.use_audio_seg:
            if self.is_valid:
                audio_seg, sr = self.get_wav(audiopath)
            else:
                audio_seg, sr = self.get_segment_wav(audiopath)
        else:
            audio_seg = None
        if self.use_audio:
            audio, sr = self.get_wav(audiopath)
            audio = torch.from_numpy(audio).float()
        else:
            audio = None
        mel = torch.from_numpy(self.get_mel(audiopath=audiopath)).float() if self.use_mel else None
        mel_seg = torch.from_numpy(self.get_mel(audio=audio_seg)).float() if self.use_mel_seg else None
        coarse_mel = torch.from_numpy(self.get_mel(audiopath, r=self.reduction_factor)).float() if self.use_coarse_mel else None
        text = self.get_text(text) if self.use_text else None
        if self.use_attn_guide:
            attn_guide, attn_mask, attn_mask2 = self.get_attn_guides_and_masks(audiopath)
        else:
            attn_guide, attn_mask = None, None
        if audio_seg is not None:
            audio_seg = torch.from_numpy(audio_seg).float()
        tvmt = self.get_time_varying_meta_template(audiopath) if self.use_tvmt else None

        return audio, audio_seg,\
               mel, mel_seg, coarse_mel, \
               text, \
               attn_guide, attn_mask, \
               tvmt, attn_mask2

    def __getitem__(self, index):
        try:
            return self.get_batch(self.audiopaths[index])
        except IndexError:
            pass

    def __len__(self):
        return len(self.audiopaths)


class _DataCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, conf):
        data_conf = conf['data']
        self.reduction_factor = data_conf['reduction_factor'] if data_conf['reduction_factor'] else None

        if 'mel' in data_conf['batch']:
            self.is_norm = data_conf['is_norm']
            self.use_mel = True
        else:
            self.use_mel = False

        self.use_audio = True if 'audio' in data_conf['batch'] else False
        self.use_audio_seg = True if 'audio_seg' in data_conf['batch'] else False
        self.use_mel_seg = True if 'mel_seg' in data_conf['batch'] else False
        self.use_coarse_mel = True if self.reduction_factor is not None and self.reduction_factor > 1 else False
        self.use_mel = True if 'mel' in data_conf['batch'] else False
        self.use_text = True if 'text' in data_conf['batch'] else False
        self.use_attn_guide = True if 'attn_guide' in data_conf['batch'] else False
        self.use_attn_mask = True if 'attn_mask' in data_conf['batch'] else False
        self.use_attn_mask2 = True if 'attn_mask2' in data_conf['batch'] else False
        self.use_tvmt = True if 'tvmt' in data_conf['batch'] else False

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [[text_normalized, mel_normalized], ...]
        """

        if self.use_text:
            # Right zero-pad all one-hot text sequences to max input length
            input_lengths, ids_sorted_decreasing = torch.sort(
                torch.LongTensor([len(x[5]) for x in batch]),
                dim=0, descending=True)
            max_input_len = input_lengths[0]

            text_padded = torch.LongTensor(len(batch), max_input_len)
            text_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                text = batch[ids_sorted_decreasing[i]][5]
                text_padded[i, :text.size(0)] = text

            num_mels = batch[0][2].size(0)
            max_target_len = max([x[2].size(1) for x in batch])
            mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
            mel_padded.zero_()
            output_lengths = torch.LongTensor(len(batch))

            if self.use_coarse_mel:
                max_coarse_target_len = max([x[4].size(1) for x in batch])
                coarse_mel_padded = torch.FloatTensor(len(batch), num_mels, max_coarse_target_len)
                coarse_mel_padded.zero_()
                output_coarse_lengths = torch.LongTensor(len(batch))
                coarse_rs = torch.FloatTensor(len(batch))

            else:
                coarse_mel_padded = None

            if self.use_attn_guide:
                attn_guide_padded = torch.FloatTensor(len(batch), max_input_len, max_coarse_target_len)
                attn_guide_padded.zero_()
                attn_mask_padded = torch.FloatTensor(len(batch), max_input_len, max_coarse_target_len)
                attn_mask_padded.zero_()
            else:
                attn_mask_padded, attn_mask_guide = None

            if self.use_attn_mask2:
                attn_mask_padded2 = torch.FloatTensor(len(batch), max_input_len, max_coarse_target_len)
                attn_mask_padded2.zero_()
            else:
                attn_mask_padded2 = None

            if self.use_tvmt:
                tvmt_padded = torch.FloatTensor(len(batch), num_mels, max_coarse_target_len)
                coarse_mel_padded.zero_()
            else:
                tvmt_padded = None

            for i in range(len(ids_sorted_decreasing)):
                mel = batch[ids_sorted_decreasing[i]][2]
                mel_padded[i, :, :mel.size(1)] = mel
                output_lengths[i] = mel.size(1)

                if self.use_coarse_mel:
                    coarse_mel = batch[ids_sorted_decreasing[i]][4]
                    coarse_mel_padded[i, :, :coarse_mel.size(1)] = coarse_mel
                    output_coarse_lengths[i] = coarse_mel.size(1)
                    coarse_rs[i] = output_coarse_lengths[i].float() / input_lengths[i].float()

                if self.use_attn_guide:
                    attn_guide = batch[ids_sorted_decreasing[i]][6][:max_input_len, :max_coarse_target_len]
                    attn_mask = batch[ids_sorted_decreasing[i]][7][:max_input_len, :max_coarse_target_len]
                    attn_guide_padded[i, :max_input_len, :attn_guide.size(1)] = attn_guide
                    attn_mask_padded[i, :max_input_len, :attn_mask.size(1)] = attn_mask

                if self.use_attn_mask2:
                    attn_mask2 = batch[ids_sorted_decreasing[i]][9][:max_input_len, :max_coarse_target_len]
                    attn_mask_padded2[i, :max_input_len, :attn_mask2.size(1)] = attn_mask2

                if self.use_tvmt:
                    tvmt = batch[ids_sorted_decreasing[i]][8]
                    tvmt_padded[i, :, :tvmt.size(1)] = tvmt


            new_batch = {
                "text": text_padded,
                "mel": mel_padded,
                "coarse_mel": coarse_mel_padded,
                "attn_mask": None, # attn_mask_padded,
                "attn_guide": attn_guide_padded,
                "ilens": input_lengths,
                "olens": output_lengths,
                "coarse_olens": output_coarse_lengths,
                "tvmt": tvmt_padded,
                "coarse_length_ratio": coarse_rs,
                "attn_mask2": attn_mask_padded2
            }

        else:
            seq_aud_len = batch[0][1].shape[0]
            audios_seg = torch.FloatTensor(len(batch), 1, seq_aud_len)

            num_mels = batch[0][3].shape[0]
            seg_mel_len = batch[0][3].shape[1]
            mels_seg = torch.FloatTensor(len(batch), num_mels, seg_mel_len - 1)
            mels_seg.zero_()

            for i in range(len(batch)):
                if self.use_audio_seg:
                    audio = batch[i][1]
                    audios_seg[i, :, :] = audio
                if self.use_mel_seg:
                    mel_seg = batch[i][3]
                    mels_seg[i, :, :] = mel_seg[:, :-1]

            new_batch = {
                "mels_seg": mels_seg,
                "audios_seg": audios_seg,
            }

        return new_batch


def guide_attention(text_lengths, mel_lengths, guide_g=0.2, r=None, c=None):
    b = len(text_lengths)
    if r is None:
        r = np.max(text_lengths)
    if c is None:
        c = np.max(mel_lengths)
    guide = np.ones((b, r, c), dtype=np.float32)
    mask = np.zeros((b, r, c), dtype=np.float32)
    for i in range(b):
        W = guide[i]
        M = mask[i]
        N = float(text_lengths[i])
        T = float(mel_lengths[i])
        for n in range(r):
            for t in range(c):
                if n < N and t < T:
                    w = 1.0 - np.exp(-(float(n) / N - float(t) / T) ** 2 / (2.0 * (guide_g ** 2)))
                    m = 1.0
                    W[n][t] = w
                    M[n][t] = m
                elif t >= T and n < N:
                    w = 1.0 - np.exp(-((float(n - N - 1) / N) ** 2 / (2.0 * (guide_g ** 2))))
                    W[n][t] = w
    return guide, mask


# class TextMelCollate():
#     """ Zero-pads model inputs and targets based on number of frames per setep
#     """
#     def __init__(self, conf, is_inference=False):
#         self.conf = conf
#         self.length_ratio_max = self.conf['data']['length_ratio_max']
#         self.length_ratio_min = self.conf['data']['length_ratio_min']
#
#         self.is_inference = is_inference
#         self.tvmt = TimeVaryingMetaTemplate(40)
#
#     def __call__(self, batch):
#         """Collate's training batch from normalized text and mel-spectrogram
#         PARAMS
#         ------
#         batch: [[text_normalized, mel_normalized], ...]
#         """

#
#             if self.conf['train']['guided_attention']:
#                 attn_guide_padded = torch.FloatTensor(len(batch), max_input_len, max_coarse_target_len)
#                 attn_guide_padded.zero_()
#                 attn_mask_padded = torch.FloatTensor(len(batch), max_input_len, max_coarse_target_len)
#                 attn_mask_padded.zero_()
#             for i in range(len(ids_sorted_decreasing)):
#                 mel = batch[ids_sorted_decreasing[i]][1]
#                 mel_padded[i, :, :mel.size(1)] = mel
#                 coarse_mel = batch[ids_sorted_decreasing[i]][2]
#                 coarse_mel_padded[i, :, :coarse_mel.size(1)] = coarse_mel
#                 _ilen = len(batch[ids_sorted_decreasing[i]][0])
#                 _olen = coarse_mel.size(1)
#                 tvmt = self.tvmt(_olen, _ilen)
#                 tvmt_padded[i, :, :coarse_mel.size(1)] = tvmt
#                 gate_padded[i, mel.size(1)-1:] = 1
#                 output_lengths[i] = mel.size(1)
#                 coarse_output_lengths[i] = coarse_mel.size(1)
#                 if self.conf['train']['guided_attention']:
#                     attn_guide = batch[ids_sorted_decreasing[i]][3][:max_input_len, :max_coarse_target_len]
#                     attn_mask = batch[ids_sorted_decreasing[i]][4][:max_input_len, :max_coarse_target_len]
#                     attn_guide_padded[i, :max_input_len, :attn_guide.size(1)] = attn_guide
#                     attn_mask_padded[i, :max_input_len, :attn_mask.size(1)] = attn_mask
#                 ids.append(batch[ids_sorted_decreasing[i]][5])
#
#             xs = text_padded
#             ilens = input_lengths
#             ys = mel_padded
#             coarse_ys = coarse_mel_padded
#             coarse_olens = coarse_output_lengths
#             olens = output_lengths
#             gates = gate_padded
#             tvmts = tvmt_padded
#             if self.conf['train']['guided_attention']:
#                 attn_masks = attn_mask_padded
#                 attn_guides = attn_guide_padded
#             else:
#                 attn_masks = None
#                 attn_guides = None
#
#             if self.length_ratio_max is not None:
#                 length_ratio = olens.float() / ilens.float()
#                 length_ratio = (length_ratio - self.length_ratio_min) / (self.length_ratio_max - self.length_ratio_min)
#             else:
#                 length_ratio = None
