from tensorboardX import SummaryWriter
from utils.plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy

class TensorboardLog(SummaryWriter):
    def __init__(self, logdir):
        super(TensorboardLog, self).__init__(logdir)

    def log_loss(self, losses, iteration, name=None):
        for loss in losses:
            for k in loss.keys():
                if name is not None:
                    self.add_scalar(name + '_' + k, loss[k], iteration)
                else:
                    self.add_scalar(k, loss[k], iteration)

    def log_spec(self, specs, olens, iteration, name=None, num=None):
        for spec in specs:
            for k in spec.keys():
                s = spec[k].transpose(1, 2)[0][:, :olens[0]].cpu().data
                if name is not None:
                    self.add_figure(name + '_' + num + '_' + k, plot_spectrogram_to_numpy(s),
                                   iteration)
                else:
                    self.add_figure(k, plot_spectrogram_to_numpy(s),
                                   iteration)

    def log_attn_ws(self, attn_ws, ilens, olens, iteration, name=None, num=None):
        for w in attn_ws:
            for k in w.keys():
                att = w[k].transpose(1, 2)[0][:olens[0], :ilens[0]].cpu().data
                if name is not None:
                    self.add_figure(name + '_' + num + '_' + k, plot_alignment_to_numpy(att),
                                    iteration)
                else:
                    self.add_figure(k, plot_alignment_to_numpy(att),
                                   iteration)

    def log_wav(self, wavs, iteration):
        for wav in wavs:
            for k in wav.keys():
                self.add_audio(k, wav[k][0], iteration, 22050)


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


# def load_wav_to_torch(full_path):
#     audio, rate = librosa.load(full_path, sr=22050)
#     return torch.FloatTensor(audio.astype(np.float32)), rate

