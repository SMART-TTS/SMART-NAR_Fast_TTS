import os
import numpy as np
from tqdm import tqdm

from utils.data_utils import _DataLoader, guide_attention
from utils.plot import plot_spectrum, plot_attention
from utils import TimeVaryingMetaTemplate


def preprocess(conf):
    print("pre-processing...")
    # validation_files training_files
    TVMT = TimeVaryingMetaTemplate(40)
    loader = _DataLoader(conf['data']['training_files'], conf)
    is_norm = conf['data']['is_norm']
    with open(conf['data']['training_files'], 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if 'chj' in conf['data']['data_name']:
                fname, txt = line.split('|')
            elif conf['data']['data_name'] == 'lj':
                fname, txt = line.split('|')
                fname = os.path.join('db', 'LJSpeech-1.1', 'wavs', '{}.wav'.format(fname))
            elif conf['data']['data_name'] == 'SMART_TTS':
                fname, txt = line.split('|')
                fname = os.path.join('db', 'SMART_TTS_20201105', 'FPHJ0', 'wav_22050', '{}.wav'.format(fname))

            mel = loader.get_mel(fname)
            text = loader.get_text(txt.rstrip())

            if 'chj' in conf['data']['data_name']:
                folder = fname.split(os.path.sep)[-2]
            name, ext = os.path.splitext(os.path.basename(fname))

            if not os.path.exists(os.path.join(conf['data']['featdir'], conf['data']['data_name'])):
                os.makedirs(os.path.join(conf['data']['featdir'], conf['data']['data_name']))

            if is_norm:
                mel_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'mels_norm')
            else:
                mel_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'mels')
            if not os.path.exists(mel_path):
                os.makedirs(mel_path)

            if 'chj' in conf['data']['data_name']:
                mel_path = os.path.join(mel_path, folder)
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

            if 'chj' in conf['data']['data_name']:
                coarse_mel_path = os.path.join(coarse_mel_path, folder)
                if not os.path.exists(coarse_mel_path):
                    os.makedirs(coarse_mel_path)

            # print(os.path.join(coarse_mel_path, name + ".npy"))
            # plot_spectrum(coarse_mel, 'coarse_mel')
            np.save(os.path.join(coarse_mel_path, name + ".npy"), coarse_mel.astype(np.float32))

            coarse_t = coarse_mel.shape[1]
            tvmt = TVMT(coarse_t, len(text))
            tvmt_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'tvmt')
            if not os.path.exists(tvmt_path):
                os.makedirs(tvmt_path)

            if 'chj' in conf['data']['data_name']:
                tvmt_path = os.path.join(tvmt_path, folder)
                if not os.path.exists(tvmt_path):
                    os.makedirs(tvmt_path)
            tvmt = tvmt.detach().cpu().numpy()
            np.save(os.path.join(tvmt_path, name + ".npy"), tvmt.astype(np.float32))

            guide, mask = guide_attention([len(text)], [coarse_mel.shape[-1]],
                                          conf['train']['guided_attention']['guide_g'],
                                          conf['data']['data_max_text_length'],
                                          conf['data']['data_max_mel_length'])

            guide_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'guide')
            if not os.path.exists(guide_path):
                os.makedirs(guide_path)

            if 'chj' in conf['data']['data_name']:
                guide_path = os.path.join(guide_path, folder)
                if not os.path.exists(guide_path):
                    os.makedirs(guide_path)

            # plot_attention(guide[0], 'guide')
            np.save(os.path.join(guide_path, name + ".npy"), guide[0].astype(np.float32))

            mask_path = os.path.join(conf['data']['featdir'], conf['data']['data_name'], 'mask')
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)

            if 'chj' in conf['data']['data_name']:
                mask_path = os.path.join(mask_path, folder)
                if not os.path.exists(mask_path):
                    os.makedirs(mask_path)

            # plot_attention(mask[0], 'mask')
            np.save(os.path.join(mask_path, name + ".npy"), mask[0].astype(np.float32))

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

if __name__ == "__main__":
    import torch
    from utils.text import text_to_sequence


    # mel = np.load('/media/tts/f31fb3cf-8485-4868-8b63-75c78781b13d/App/kss_v_1.4-mel-4_5628.npy')
    # plot_spectrum(mel, '123123123')

    # make attn_mask for attn masking
    # with open('/media/tts/f31fb3cf-8485-4868-8b63-75c78781b13d/App/asmltts_beta/db/SMART_TTS_20201105/train.txt', 'r') as f:
    #     lines = f.readlines()
    #     for line in tqdm(lines):
    #         fname, txt = line.split('|')
    #         mel_path = os.path.join('feats/SMART_TTS/coarse_mels_norm', fname + '.npy')
    #         mel = np.load(mel_path)
    #         T = mel.shape[1]
    #         text = torch.IntTensor(text_to_sequence(txt.rstrip()))
    #         N = len(text)
    #
    #         mask = attn_mask([N], [T], 0.15, 400, 480)
    #
    #         mask_path = mel_path.replace('coarse_mels_norm', 'mask_for_attn_masking')
    #         np.save(mask_path, mask[0])




    # with open('/media/tts/f31fb3cf-8485-4868-8b63-75c78781b13d/App/asmltts_beta/db/LJSpeech-1.1/train.txt', 'w') as f2:
    #     with open('/media/tts/f31fb3cf-8485-4868-8b63-75c78781b13d/App/asmltts_beta/db/LJSpeech-1.1/metadata.csv',
    #               'r') as f:
    #         lines = f.readlines()
    #         for line in tqdm(lines):
    #             fname, _, txt = line.split('|')
    #             f2.write(fname + '|' + txt)

    # get rs
    with open('db/SMART_TTS_20201105/train.txt', 'r') as f:
        lines = f.readlines()
        rs_min = 50
        rs_max = 0
        rs_list = []
        for line in tqdm(lines):
            fname, txt = line.split('|')

            audiopath_and_text = [fname, txt.rstrip()]
            text = torch.IntTensor(text_to_sequence(txt.rstrip()))
            # mel_path = fname.replace('db/chj_22050', 'feats/chj_22050/coarse_mels_norm')
            mel_path = os.path.join('feats', 'SMART_TTS', 'coarse_mels_norm', fname + '.npy')
            # mel_path = mel_path.replace('wav', 'npy')
            mel = np.load(mel_path)

            len_mel = mel.shape[1]
            rs = float(len_mel / len(text))
            rs_min = rs if rs < rs_min else rs_min
            rs_max = rs if rs > rs_max else rs_max
            # rs_list.append(rs)

        print(rs_min, rs_max)

    # Trimming
    # import librosa
    # with open('db/chj_22050/validation.txt', 'r') as f:
    #     lines = f.readlines()
    #     for line in tqdm(lines):
    #         fname, txt = line.split('|')
    #         # fname_in = fname
    #         fname_in = fname.replace('_trim.wav', '.wav')
    #         audio, rate = librosa.load(fname_in, sr=22050)
    #         audio, _ = librosa.effects.trim(audio, top_db=20)
    #         # audio = (audio * 32767).astype(int)
    #
    #         out_name = fname
    #         # out_name = fname.replace('.wav', '_trim.wav')
    #         librosa.output.write_wav(out_name, audio, rate)
    #         # scipy.io.wavfile.write(out_name, 22050, audio)

    # resample
    import librosa
    with open('/media/tts/f31fb3cf-8485-4868-8b63-75c78781b13d/App/db/SMART_TTS_20201105/train.txt', 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            fname, txt = line.split('|')
            fname_in = os.path.join('db/SMART_TTS_20201105/FPHJ0/wav_48000', fname + '.wav')
            audio, rate = librosa.load(fname_in, sr=48000)
            audio_ = librosa.resample(audio, rate, 22050)
            # audio = (audio * 32767).astype(int)

            out_name = fname_in.replace('wav_48000', 'wav_22050')

            # out_name = fname.replace('.wav', '_trim.wav')
            librosa.output.write_wav(out_name, audio_, 22050)
            # scipy.io.wavfile.write(out_name, 22050, audio)
