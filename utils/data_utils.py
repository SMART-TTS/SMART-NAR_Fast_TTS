import os
import re
import random
import librosa
import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F

from utils import load_filepaths_and_text
from utils.text import text_to_sequence
from utils.spectrogram import logmelspec


class _DataLoader(torch.utils.data.Dataset):
    def __init__(self, paths, conf, valid=False):
        data_conf = conf['data']
        self.is_valid = valid
        self.data_name = data_conf['data_name']

        self.use_audio = True if 'audio' in data_conf['batch'] else False
        self.use_audio_seg = True if 'audio_seg' in data_conf['batch'] else False
        self.use_mel_seg = True if 'mel_seg' in data_conf['batch'] else False
        self.use_mel = True if 'mel' in data_conf['batch'] else False
        self.use_coarse_mel = True if 'coarse_mel' in data_conf['batch'] else False
        self.use_text = True if 'text' in data_conf['batch'] else False
        self.use_attn_guide = True if 'attn_guide' in data_conf['batch'] else False
        self.use_attn_mask = True if 'attn_mask' in data_conf['batch'] else False

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

        self.audiopath_and_text = load_filepaths_and_text(paths)

        if not valid:
            random.seed(1234)
            random.shuffle(self.audiopath_and_text)

    def get_batch(self, audiopath_and_text):
        if len(audiopath_and_text) > 1:
            audiopath = audiopath_and_text[0]
            text = audiopath_and_text[1]
            if self.data_name == 'SMART_TTS':
                audiopath = os.path.join('db', 'SMART_TTS', 'FPHJ0', 'wav_22050', audiopath + '.wav')

        # inference only
        else:
            audiopath = None
            text = audiopath_and_text[0]

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
        mel_seg = torch.from_numpy(logmelspec(audiopath=None, sampling_rate=self.sampling_rate,
                                              n_mel=self.n_mel, n_fft=self.n_fft,
                                              hop_length=self.hop_length, win_length=self.win_length,
                                              audio_refdB=self.audio_refdB, audio_maxdB=self.audio_maxdB,
                                              audio=audio_seg)).float() if self.use_mel_seg else None
        coarse_mel = torch.from_numpy(self.get_mel(audiopath, r=self.reduction_factor)).float() if self.use_coarse_mel else None
        text = self.get_text(text) if self.use_text else None

        if self.use_attn_guide:
            attn_guide, attn_mask = self.get_attn_guides_and_masks(audiopath)
        else:
            attn_guide, attn_mask = None, None

        audio_seg = torch.from_numpy(audio_seg).float() if audio_seg is not None else None

        return audio, audio_seg,\
               mel, mel_seg, coarse_mel, \
               text, \
               attn_guide, attn_mask


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
        return audio, rate

    def get_mel(self, audiopath=None, r=1):
        if r == 1:
            if 'chj' in self.data_name:
                mel_path = audiopath.replace('db/{}'.format(self.data_name), 'feats/{}/mels'.format(self.data_name))
                mel_path = mel_path.replace('wav', 'npy')
            elif self.data_name == 'lj':
                mel_path = os.path.join('feats', 'lj', 'mels', '{}.npy'.format(audiopath))
            elif self.data_name == 'SMART_TTS':
                mel_path = audiopath.replace('db/SMART_TTS', 'feats/{}/mels'.format(self.data_name))
                mel_path = re.sub("...../wav_22050/", "", mel_path)
                mel_path = mel_path.replace('.wav', '.npy')
            mel = np.load(mel_path)

        else:
            if 'chj' in self.data_name:
                mel_path = audiopath.replace('db/{}'.format(self.data_name), 'feats/{}/coarse_mels'.format(self.data_name))
                mel_path = mel_path.replace('wav', 'npy')
            elif self.data_name == 'lj':
                mel_path = os.path.join('feats', 'lj', 'coarse_mels', '{}.npy'.format(audiopath))
            elif self.data_name == 'SMART_TTS':
                mel_path = audiopath.replace('db/SMART_TTS', 'feats/{}/coarse_mels'.format(self.data_name))
                mel_path = re.sub("...../wav_22050/", "", mel_path)
                mel_path = mel_path.replace('.wav', '.npy')
            mel = np.load(mel_path)
        return mel

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text.rstrip()))
        return text_norm

    def get_attn_guides_and_masks(self, audiopath):
        if 'chj' in self.data_name:
            guide_path = audiopath.replace('db/{}'.format(self.data_name), 'feats/{}/guide'.format(self.data_name))
            guide_path = guide_path.replace('wav', 'npy')
            mask_path = audiopath.replace('db/{}'.format(self.data_name), 'feats/{}/mask'.format(self.data_name))
            mask_path = mask_path.replace('wav', 'npy')
        elif self.data_name == 'lj':
            guide_path = os.path.join('feats', 'lj', 'guide', '{}.npy'.format(audiopath))
            mask_path = os.path.join('feats', 'lj', 'mask', '{}.npy'.format(audiopath))
        elif self.data_name == 'SMART_TTS':
            guide_path = audiopath.replace('db/SMART_TTS', 'feats/{}/guide'.format(self.data_name))
            guide_path = re.sub("...../wav_22050/", "", guide_path)
            guide_path = guide_path.replace('.wav', '.npy')
            mask_path = audiopath.replace('db/SMART_TTS', 'feats/{}/mask'.format(self.data_name))
            mask_path = re.sub("...../wav_22050/", "", mask_path)
            mask_path = mask_path.replace('.wav', '.npy')
        attn_guide = np.load(guide_path)
        attn_mask = np.load(mask_path)

        return (torch.from_numpy(attn_guide), torch.from_numpy(attn_mask))

    def __getitem__(self, index):
        try:
            return self.get_batch(self.audiopath_and_text[index])
        except IndexError:
            pass

    def __len__(self):
        return len(self.audiopath_and_text)


class _DataCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, conf, valid=False):
        data_conf = conf['data']
        self.is_valid = valid

        self.reduction_factor = data_conf['reduction_factor'] if data_conf['reduction_factor'] else None

        self.use_text = True if 'text' in data_conf['batch'] else False
        self.use_audio = True if 'audio' in data_conf['batch'] else False
        self.use_audio_seg = True if 'audio_seg' in data_conf['batch'] else False
        self.use_mel_seg = True if 'mel_seg' in data_conf['batch'] else False
        self.use_mel = True if 'mel' in data_conf['batch'] else False
        self.use_coarse_mel = True if 'coarse_mel' in data_conf['batch'] else False
        self.use_attn_guide = True if 'attn_guide' in data_conf['batch'] else False
        self.use_attn_mask = True if 'attn_mask' in data_conf['batch'] else False

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [[text_normalized, mel_normalized], ...]
        """

        ids_sorted_decreasing = None
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
        else:
            text_padded, input_lengths = None, None

        if self.use_audio_seg:
            seq_aud_len = batch[0][1].shape[0]
            audio_seg = torch.FloatTensor(len(batch), 1, seq_aud_len)
        else:
            audio_seg = None

        if self.use_mel_seg:
            num_mels = batch[0][3].shape[0]
            seg_mel_len = batch[0][3].shape[1]
            mels_seg = torch.FloatTensor(len(batch), num_mels, seg_mel_len - 1)
            mels_seg.zero_()
        else:
            mels_seg = None

        if self.use_mel:
            num_mels = batch[0][2].size(0)
            max_target_len = max([x[2].size(1) for x in batch])
            mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
            mel_padded.zero_()
            output_lengths = torch.LongTensor(len(batch))
        else:
            mel_padded, output_lengths = None, None

        if self.use_coarse_mel:
            max_coarse_target_len = max([x[4].size(1) for x in batch])
            coarse_mel_padded = torch.FloatTensor(len(batch), num_mels, max_coarse_target_len)
            coarse_mel_padded.zero_()
            output_coarse_lengths = torch.LongTensor(len(batch))
            coarse_rs = torch.FloatTensor(len(batch))

        else:
            coarse_mel_padded, output_coarse_lengths, coarse_rs = None, None, None

        if self.use_attn_guide:
            attn_guide_padded = torch.FloatTensor(len(batch), max_input_len, max_coarse_target_len)
            attn_guide_padded.zero_()
        else:
            attn_mask_padded = None

        if self.use_attn_mask:
            attn_mask_padded = torch.FloatTensor(len(batch), max_input_len, max_coarse_target_len)
            attn_mask_padded.zero_()
        else:
            attn_guide_padded = None

        if ids_sorted_decreasing is not None:
            for i in range(len(ids_sorted_decreasing)):
                if self.use_mel:
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
                    attn_guide_padded[i, :max_input_len, :attn_guide.size(1)] = attn_guide

                if self.use_attn_mask:
                    attn_mask = batch[ids_sorted_decreasing[i]][7][:max_input_len, :max_coarse_target_len]
                    attn_mask_padded[i, :max_input_len, :attn_mask.size(1)] = attn_mask

        else:
            for i in range(len(batch)):
                if self.use_audio_seg:
                    audio = batch[i][1]
                    audio_seg[i, :, :] = audio
                if self.use_mel_seg:
                    _mel_seg = batch[i][3]
                    mels_seg[i, :, :] = _mel_seg[:, :-1]

        new_batch = {
            "text": text_padded,
            "mel": mel_padded,
            "coarse_mel": coarse_mel_padded,
            "attn_mask": attn_mask_padded,
            "attn_guide": attn_guide_padded,
            "ilens": input_lengths,
            "olens": output_lengths,
            "coarse_olens": output_coarse_lengths,
            "audios_seg": audio_seg,
            "mels_seg": mels_seg
        }

        return new_batch

