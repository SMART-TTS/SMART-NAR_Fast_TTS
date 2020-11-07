import os
import numpy as np
from tqdm import tqdm

from utils.data_utils import _DataLoader, guide_attention


def preprocess(conf):
    print("pre-processing...")
    loader = _DataLoader(conf['data']['training_files'], conf)
    is_norm = conf['data']['is_norm']
    with open(conf['data']['training_files'], 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            fname, txt = line.split('|')
            fname = os.path.join('db', 'SMART_TTS_20201105', 'FPHJ0', 'wav_22050', '{}.wav'.format(fname))

            mel = loader.get_mel(fname)
            text = loader.get_text(txt.rstrip())

            name, ext = os.path.splitext(os.path.basename(fname))

            if not os.path.exists(os.path.join(conf['data']['featdir'], conf['data']['data_name'])):
                os.makedirs(os.path.join(conf['data']['featdir'], conf['data']['data_name']))

            if is_norm:
                mel_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'mels_norm')
            else:
                mel_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'mels')
            if not os.path.exists(mel_path):
                os.makedirs(mel_path)

            # plot_spectrum(mel, 'mel')
            np.save(os.path.join(mel_path, name + ".npy"), mel.astype(np.float32))

            t = mel.shape[1]
            pad = conf['data']['reduction_factor'] - (t % conf['data']['reduction_factor']) if t % conf['data']['reduction_factor']  != 0 else 0
            coarse_mel = np.pad(mel, [[0, 0], [0, pad]], mode="constant")
            coarse_mel = coarse_mel[..., ::conf['data']['reduction_factor']]

            if is_norm:
                coarse_mel_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'coarse_mels_norm')
            else:
                coarse_mel_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'coarse_mels')
            if not os.path.exists(coarse_mel_path):
                os.makedirs(coarse_mel_path)

            # plot_spectrum(coarse_mel, 'coarse_mel')
            np.save(os.path.join(coarse_mel_path, name + ".npy"), coarse_mel.astype(np.float32))

            guide, mask = guide_attention([len(text)], [coarse_mel.shape[-1]],
                                          conf['train']['guided_attention']['guide_g'],
                                          conf['data']['data_max_text_length'],
                                          conf['data']['data_max_mel_length'])

            guide_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'guide')
            if not os.path.exists(guide_path):
                os.makedirs(guide_path)
            # plot_attention(guide[0], 'guide')
            np.save(os.path.join(guide_path, name + ".npy"), guide[0].astype(np.float32))

            mask_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'mask')
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)
            # plot_attention(mask[0], 'mask')
            np.save(os.path.join(mask_path, name + ".npy"), mask[0].astype(np.float32))

            mask2 = attn_mask([len(text)], [coarse_mel.shape[-1]], 0.15,
                              conf['data']['data_max_text_length'],
                              conf['data']['data_max_mel_length'])
            mask2_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'mask_for_attn_masking')
            if not os.path.exists(mask2_path):
                os.makedirs(mask2_path)
            np.save(os.path.join(mask2_path, name + ".npy"), mask2[0].astype(np.float32))
            np.save(mask2_path, mask2[0])


def attn_mask(text_lengths, mel_lengths, guide_g=0.2, r=None, c=None):
    b = len(text_lengths)
    if r is None:
        r = np.max(text_lengths)
    if c is None:
        c = np.max(mel_lengths)
    mask = np.ones((b, r, c), dtype=np.float32)
    for i in range(b):
        W = mask[i]
        N = float(text_lengths[i])
        T = float(mel_lengths[i])
        for n in range(r):
            for t in range(c):
                if n < N and t < T:
                    w = np.exp(-(float(n) / N - float(t) / T) ** 2 / (2.0 * (guide_g ** 2)))
                    W[n][t] = w
                elif t >= T and n < N:
                    w = np.exp(-((float(n - N - 1) / N) ** 2 / (2.0 * (guide_g ** 2))))
                    W[n][t] = w
    return mask
