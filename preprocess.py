import os
import yaml
import librosa
import argparse
import numpy as np
from tqdm import tqdm

import torch

from utils.text import text_to_sequence
from utils.mask import get_guide_and_mask_for_attention
from utils.spectrogram import logmelspec
from utils.plot import plot_spectrum, plot_attention


def preprocess(args):
    conf = yaml.load(open(args.conf))
    conf = conf['data']

    sampling_rate = conf['sampling_rate']
    n_mel = conf['n_mel']
    n_fft = conf['n_fft']
    hop_length = conf['hop_length']
    win_length = conf['win_length']
    audio_refdB = conf['audio_refdB']
    audio_maxdB = conf['audio_maxdB']

    print("pre-processing...")

    for file in [conf['training_files'], conf['validation_files']]:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                if 'chj' in conf['data_name']:
                    fname, txt = line.split('|')
                    folder = fname.split(os.path.sep)[-2]
                elif conf['data_name'] == 'lj':
                    fname, txt = line.split('|')
                    fname = os.path.join('db', 'LJSpeech-1.1', 'wavs', '{}.wav'.format(fname))
                elif conf['data_name'] == 'SMART_TTS':
                    fname, txt = line.split('|')
                    speaker = fname.split('_')[0]
                    fname = os.path.join('db', 'SMART_TTS', speaker, 'wav_22050', '{}.wav'.format(fname))

                name, ext = os.path.splitext(os.path.basename(fname))
                if not os.path.exists(os.path.join(conf['featdir'], conf['data_name'])):
                    os.makedirs(os.path.join(conf['featdir'], conf['data_name']))

                mel = logmelspec(fname, sampling_rate, n_mel, n_fft, hop_length, win_length, audio_refdB, audio_maxdB)
                text = torch.IntTensor(text_to_sequence(txt.rstrip()))

                mel_path = os.path.join(conf['featdir'], conf['data_name'], 'mels')
                if not os.path.exists(mel_path):
                    os.makedirs(mel_path)
                if 'chj' in conf['data_name']:
                    mel_path = os.path.join(mel_path, folder)
                    if not os.path.exists(mel_path):
                        os.makedirs(mel_path)

                np.save(os.path.join(mel_path, name + ".npy"), mel.astype(np.float32))
                # plot_spectrum(mel, 'mel')

                t = mel.shape[1]
                pad = conf['reduction_factor'] - (t % conf['reduction_factor']) if t % conf['reduction_factor']  != 0 else 0
                coarse_mel = np.pad(mel, [[0, 0], [0, pad]], mode="constant")
                coarse_mel = coarse_mel[..., ::conf['reduction_factor']]

                coarse_mel_path = os.path.join(conf['featdir'], conf['data_name'], 'coarse_mels')
                if not os.path.exists(coarse_mel_path):
                    os.makedirs(coarse_mel_path)
                if 'chj' in conf['data_name']:
                    coarse_mel_path = os.path.join(coarse_mel_path, folder)
                    if not os.path.exists(coarse_mel_path):
                        os.makedirs(coarse_mel_path)
                # np.save(os.path.join(coarse_mel_path, name + ".npy"), coarse_mel.astype(np.float32))

                guide, mask = get_guide_and_mask_for_attention([len(text)], [coarse_mel.shape[-1]],
                                                               conf['guided_attention']['g'],
                                                               conf['attention_masking']['g'],
                                                               conf['data_max_text_length'],
                                                               conf['data_max_mel_length'])

                guide_path = os.path.join(conf['featdir'], conf['data_name'], 'guide')
                if not os.path.exists(guide_path):
                    os.makedirs(guide_path)
                if 'chj' in conf['data_name']:
                    guide_path = os.path.join(guide_path, folder)
                    if not os.path.exists(guide_path):
                        os.makedirs(guide_path)
                np.save(os.path.join(guide_path, name + ".npy"), guide[0].astype(np.float32))
                # plot_attention(guide[0], 'guide')

                mask_path = os.path.join(conf['featdir'], conf['data_name'], 'mask')
                if not os.path.exists(mask_path):
                    os.makedirs(mask_path)
                if 'chj' in conf['data_name']:
                    mask_path = os.path.join(mask_path, folder)
                    if not os.path.exists(mask_path):
                        os.makedirs(mask_path)
                # np.save(os.path.join(mask_path, name + ".npy"), mask[0].astype(np.float32))
                # plot_attention(mask[0], 'mask')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', type=str,
                        default='model/tts/dcgantts/conf/dcgantts_v8.yaml',
                        help='config file path')
    args = parser.parse_args()
    preprocess(args)
