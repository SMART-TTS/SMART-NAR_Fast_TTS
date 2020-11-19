import os
import argparse
import importlib
import yaml
from tqdm import tqdm
import scipy
import librosa
from librosa.output import write_wav as write

from torch.utils.data import DataLoader
from utils.data_utils import _DataCollate, _DataLoader
from model.tts.train import load_model as load_tts_model, load_checkpoint
from model.voc.train import load_model as load_voc_model

from utils import TensorboardLog
from utils.plot import plot_spectrum, plot_attention
from utils.text import get_symbols


def prepare_dataloaders(conf):
    # Get data, data loaders and collate function ready
    testset = _DataLoader(conf['test_files'], conf, valid=True)
    collate_fn = _DataCollate(conf, valid=True)
    data_loader = DataLoader(testset, num_workers=1,
                            shuffle=False, batch_size=1,
                            pin_memory=False, collate_fn=collate_fn)

    return data_loader, collate_fn


def synthesis(args):
    conf = yaml.load(open(args.conf))
    tts_model_name = conf['model_tts']
    voc_model_name = conf['model_voc']
    tts_conf = yaml.load(open(conf['tts_conf']))
    voc_conf = yaml.load(open(conf['voc_conf']))
    device = conf['device']

    if conf['data']['text_cleaners'] == ['english_cleaners']:
        sym_to_id, _ = get_symbols('english_cleaners')
    else:
        sym_to_id, _ = get_symbols('korean_cleaners')

    tts_conf['model']['idim'] = len(sym_to_id)

    tts, _, _ = load_tts_model(tts_model_name, tts_conf, is_training=False)
    tts, _, _, _ = load_checkpoint(conf['checkpoint_tts'], tts)
    tts.eval().to(device)

    voc, _, _ = load_voc_model(voc_model_name, voc_conf, is_training=False)
    voc, _, _, _ = load_checkpoint(conf['checkpoint_voc'], voc)
    voc.to(device)
    voc.eval()

    # logger
    model_name = "{}_{}".format(os.path.splitext(os.path.basename(conf['tts_conf']))[0],
                                os.path.splitext(os.path.basename(conf['voc_conf']))[0])
    out_dir = os.path.join('decode', model_name)
    # make directories for save
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    tensorboard_dir = os.path.join(out_dir, 'logs')
    # logger = TensorboardLog(tensorboard_dir)

    data_loader, collate_fn = prepare_dataloaders(conf)

    bar = tqdm(data_loader)
    for i, batch in enumerate(bar):
        mels = batch['mel']
        ilens = batch['ilens']
        olens = batch['coarse_olens']
        print(i, ilens[0], olens[0], float(olens[0])/float(ilens[0]))
        mels_pred, attn_ws = tts(batch=batch, device='cpu')
        wav = voc(pred_mels=mels_pred[1], device='cpu')

        plot_spectrum(mels_pred[1][0].detach().numpy(), '{}_mel_pred'.format(i), dir=out_dir)
        plot_attention(attn_ws[0].detach().numpy(), '{}_attn_ws'.format(i), dir=out_dir)
        audio = (wav * 32768.0).astype("int16")
        scipy.io.wavfile.write(os.path.join(out_dir, '{}_syn.wav'.format(i)), 22050, audio[0])

        if mels is not None:
            plot_spectrum(mels[0].detach().numpy(), '{}_mel_true'.format(i), dir=out_dir)
            wav = voc(pred_mels=mels, device='cpu')
            audio = (wav * 32768.0).astype("int16")
            scipy.io.wavfile.write(os.path.join(out_dir, '{}_ground_truth.wav'.format(i)), 22050, audio[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str,
                        default='decode/conf/decode_v1.yaml',
                        help='config file path for synthesis')
    args = parser.parse_args()

    synthesis(args)
