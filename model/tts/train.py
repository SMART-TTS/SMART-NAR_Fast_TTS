import os
import yaml
import random
import importlib
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import TensorboardLog
from utils.data_utils import _DataCollate, _DataLoader

from utils.text import get_symbols
# from utils.text.symbols import eng_symbols, kor_symbols

def save_checkpoint(model, optimizers, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    state = {"iteration": iteration,
             "state_dict": model.state_dict(),
             "learning_rate": learning_rate}
    for k in optimizers.keys():
        state[k] = optimizers[k].state_dict()

    torch.save(state, filepath)


def load_checkpoint(checkpoint_path, model, optimizers=None):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'], strict=False)
    if optimizers is not None:
        for k in optimizers:
            optimizers[k].load_state_dict(checkpoint_dict[k])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizers, learning_rate, iteration


def load_model(model, conf, is_training=True):
    model = 'model.tts.' + model + '.module.' + model
    m = importlib.import_module(model)
    device = conf['train']['device']
    model = m.Model(conf, is_training).to(device)
    optimizers = m.optimizer(conf, model)

    # parallel
    # --------

    global_step = 0
    if conf['train']['checkpoint']:
        model, optimizers, learning_rate, global_step = load_checkpoint(
            conf['train']['checkpoint'], model, optimizers)

    return model, optimizers, global_step


def prepare_dataloaders(conf):
    data_conf = conf['data']
    # Get data, data loaders and collate function ready

    trainset = _DataLoader(data_conf['training_files'], conf)
    valset = _DataLoader(data_conf['validation_files'], conf, valid=True)
    collate_fn = _DataCollate(conf)

    train_sampler = DistributedSampler(trainset) \
        if conf['train']['distributed_run'] else None

    train_loader = DataLoader(trainset, num_workers=1, shuffle=True,
                              sampler=train_sampler,
                              batch_size=conf['train']['batch_size'], pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(valset, sampler=train_sampler, num_workers=1,
                            shuffle=False, batch_size=1,
                            pin_memory=False, collate_fn=collate_fn)

    return train_loader, val_loader, collate_fn


def validate(conf, model, data_loader, output_directory, logger, gs, epoch, dynamic_guide, validation_name='valid'):
    device = conf['train']['device']
    adversarial_training = conf['train']['adversarial_training']
    model.eval()
    dataiter = iter(data_loader)
    val_recon_loss = []
    with torch.no_grad():
        if not adversarial_training:
            for i in range(int(conf['train']['valid_num'])):
                batch = next(dataiter)
                loss, report_loss_keys = model(batch=batch, dynamic_guide=dynamic_guide,
                             logger=logger, gs=gs, valid_num=i, valid=True, device=device, report_name_for_outs=validation_name)
                # if len(report_loss_keys) > 15:
                #     val_recon_loss.append(report_loss_keys[-1]['loss_recon_mels'])
                # else:
                #     val_recon_loss.append(report_loss_keys[0]['loss_recon'])

            # report_keys = [
            #     {"loss_recon_mels": (sum(val_recon_loss) / len(val_recon_loss))},
            # ]
            # logger.log_loss(report_keys, gs, validation_name)
        else:
            for i in range(int(conf['train']['valid_num'])):
                batch = next(dataiter)
                _, _, _, _ = model(step='g', batch=batch, dynamic_guide=dynamic_guide,
                             logger=logger, gs=gs, valid_num=i, valid=True, device=device, report_name_for_outs=validation_name)

    model.train()


def train(args):

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    # load model
    conf = yaml.load(open(args.conf))
    if conf['preprocess']:
        from preprocess import preprocess
        preprocess(conf)

    conf['load_mel_from_disk'] = True
    if conf['data']['text_cleaners'] == ['english_cleaners']:
        sym_to_id, _ = get_symbols('english_cleaners')
    else:
        sym_to_id, _ = get_symbols('korean_cleaners')

    conf['model']['idim'] = len(sym_to_id)
    # model, optimizers, gs, scheduler = load_model(args.model, conf)
    model, optimizers, gs = load_model(args.model, conf)

    if conf['train']['ngpu'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(conf['train']['ngpu'])))
    else:
        device = conf['train']['device']

    train_loader, val_loader, collate_fn = prepare_dataloaders(conf)

    adversarial_training = conf['train']['adversarial_training']
    if adversarial_training:
        optimizer_g = optimizers['optimizer_g']
        optimizer_d = optimizers['optimizer_d']
    else:
        optimizer = optimizers['optimizer']

    start_ep = int(gs / len(train_loader)) if gs > 0 else 0
    if conf['train']['guided_attention']:
        guided_attention_conf = conf['train']['guided_attention']
        dynamic_guide = float(guided_attention_conf['guide_weight'])
        dynamic_guide *= guided_attention_conf['guide_decay'] ** gs
    else:
        dynamic_guide = None

    # logger
    exp_name = conf['train']['exp_name']
    model_name = os.path.basename(os.path.splitext(args.conf)[0])
    tensorboard_dir = os.path.join(conf['train']['tensorboard_dir'], model_name, exp_name)
    logger = TensorboardLog(tensorboard_dir)

    # make directories for save
    exp_directory = conf['train']['exp_directory']
    output_directory = os.path.join(exp_directory, model_name)
    if not os.path.exists(exp_directory):
        os.makedirs(exp_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_directory = os.path.join(exp_directory, model_name, exp_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    model.train()
    for epoch in range(start_ep, conf['train']['epoch']):
        bar = tqdm(train_loader)

        for i, batch in enumerate(bar):
            # validate(conf, model, val_loader, output_directory, logger, gs, epoch, dynamic_guide, validation_name='valid')
            # validate(conf, model, train_loader, output_directory, logger, gs, epoch, dynamic_guide, validation_name='train')

            if adversarial_training:
                loss, _, d_in_fake, d_in_real = model(step='g', batch=batch, dynamic_guide=dynamic_guide, logger=logger, gs=gs, device=device)
                optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()

                loss, _, _, _ = model(step='d', d_in_fake=d_in_fake, d_in_real=d_in_real, logger=logger, gs=gs, device=device)
                optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_d.step()

            else:
                loss, _ = model(batch=batch, dynamic_guide=dynamic_guide, logger=logger, gs=gs, device=device)

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ')
                    exit(1)

                optimizer.zero_grad()
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 1)
                optimizer.step()
                if conf['train']['guided_attention']:
                    dynamic_guide *= conf['train']['guided_attention']['guide_decay']
                    if dynamic_guide < conf['train']['guided_attention']['guide_lowbound']:
                        dynamic_guide = conf['train']['guided_attention']['guide_lowbound']

            # bar.set_description("ep: {}, lr: {}, gs: {}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr'], gs))
            bar.set_description("ep: {}, gs: {}".format(epoch, gs))
            gs += 1

        # if scheduler is not None:
        #     scheduler.step()

        if (epoch + 1) % int(conf['train']['valid_epoch']) == 0:
            validate(conf, model, val_loader, output_directory, logger, gs, epoch, dynamic_guide, validation_name='valid')
            validate(conf, model, train_loader, output_directory, logger, gs, epoch, dynamic_guide, validation_name='train')

        if (epoch + 1) % int(conf['train']['save_epoch']) == 0:
            checkpoint_path = os.path.join(
                output_directory, "checkpoint_{}".format(gs))
            if adversarial_training:
                save_checkpoint(model,
                                {"optimizer_g": optimizer_g,
                                 "optimizer_d": optimizer_d},
                                float(conf['optimizer']['adam_alpha']),
                                gs, checkpoint_path)

            else:
                save_checkpoint(model,
                                {"optimizer": optimizer},
                                float(conf['optimizer']['adam_alpha']),
                                gs, checkpoint_path)