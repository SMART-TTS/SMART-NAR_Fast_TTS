import os
import argparse
import importlib

def tts_train(model_name, args):
    args.conf = os.path.join('model', 'tts', model_name, 'conf', args.conf)
    _train = 'model.tts.train'
    m = importlib.import_module(_train)
    m.train(args)


def voc_train(model_name, args):
    args.conf = os.path.join('model', 'voc', model_name, 'conf', args.conf)
    _train = 'model.voc.train'
    m = importlib.import_module(_train)
    m.train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='tts',
                        help='tts or voc')
    parser.add_argument('-m', '--model', type=str, default='dcgantts',
                        help='model name for train or synthesis')
    parser.add_argument('-v', '--vocoder', type=str, default='melgan')
    parser.add_argument('-c', '--conf', type=str,
                        default='dcgantts_v1.yaml',
                        help='config file path')
    args = parser.parse_args()

    if args.stage == 'tts':
        tts_train(args.model, args)
    elif args.stage == 'voc':
        voc_train(args.model, args)
