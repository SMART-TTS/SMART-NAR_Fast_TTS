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
from model.voc.train import load_model as load_model_voc


def save_checkpoint(model, optimizers, schedulers, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    state = {"iteration": iteration,
             "state_dict": model.state_dict(),
             "learning_rate": learning_rate}
    for k in optimizers.keys():
        state[k] = optimizers[k].state_dict()
    if schedulers is not None:
        for k in schedulers.keys():
            state[k] = schedulers[k].state_dict()

    torch.save(state, filepath)


def load_checkpoint(checkpoint_path, model, optimizers=None, schedulers=None):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'], strict=False)
    if optimizers is not None:
        for k in optimizers:
            optimizers[k].load_state_dict(checkpoint_dict[k])
    if schedulers is not None:
        for k in schedulers:
            schedulers[k].load_state_dict(checkpoint_dict[k])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizers, schedulers, learning_rate, iteration


def load_model(model, conf, is_training=True):
    model = 'model.tts.' + model + '.module.' + model
    m = importlib.import_module(model)
    model = m.Model(conf, is_training)

    if is_training:
        optimizers, schedulers = m.optimizer(conf, model)
    else:
        optimizers, schedulers = None, None

    # parallel
    if len(list(conf['train']['device'])) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(conf['train']['device']))
        model = model.cuda()
    else:
        device = conf['train']['device']
        torch.cuda.set_device(device[0])  # change allocation of current GPU
        model = model.cuda()
        print('Current cuda device ', torch.cuda.current_device())  # check

    global_step = 0
    if is_training and conf['train']['checkpoint']:
        model, optimizers, schedulers, learning_rate, global_step = load_checkpoint(
            conf['train']['checkpoint'], model, optimizers, schedulers)

    return model, optimizers, schedulers, global_step


def prepare_dataloaders(conf):
    data_conf = conf['data']
    # Get data, data loaders and collate function ready

    trainset = _DataLoader(data_conf['training_files'], conf)
    valset = _DataLoader(data_conf['validation_files'], conf, valid=True)
    collate_fn = _DataCollate(conf)
    collate_fn_valid = _DataCollate(conf, valid=True)

    train_sampler = DistributedSampler(trainset) \
        if conf['train']['distributed_run'] else None

    train_loader = DataLoader(trainset, num_workers=1, shuffle=True,
                              sampler=train_sampler,
                              batch_size=conf['train']['batch_size'], pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(valset, sampler=train_sampler, num_workers=1,
                            shuffle=False, batch_size=1,
                            pin_memory=False, collate_fn=collate_fn_valid)

    return train_loader, val_loader, collate_fn


def validate(conf, model, batch, logger, gs, dynamic_guide, validation_name='valid', valid_num=0, voc=None):
    adversarial_training = conf['train']['adversarial_training']
    model.eval()
    with torch.no_grad():
        if not adversarial_training:
            _ = model(batch=batch,
                      dynamic_guide=dynamic_guide, voc=voc,
                      logger=logger, gs=gs, epoch=0,
                      valid_num=valid_num, valid=True, report_name_for_outs=validation_name)
        else:
            _ = model(step='g', batch=batch,
                      dynamic_guide=dynamic_guide, voc=voc,
                      logger=logger, gs=gs, epoch=0,
                      valid_num=valid_num, valid=True, report_name_for_outs=validation_name)

    model.train()


def train(args):

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    # load model
    conf = yaml.load(open(args.conf))
    if conf['data']['text_cleaners'] == ['english_cleaners']:
        sym_to_id, _ = get_symbols('english_cleaners')
    else:
        sym_to_id, _ = get_symbols('korean_cleaners')

    conf['model']['idim'] = len(sym_to_id)
    model, optimizers, schedulers, gs = load_model(args.model, conf)
    if conf['voc']:
        voc_conf = yaml.load(open(conf['voc']['conf']))
        voc_conf['train']['device'] = 'cuda:3'
        voc, _, _ = load_model_voc(conf['voc']['model'], voc_conf, is_training=False)
        voc = voc.to('cuda:3')
        voc.eval()
    else:
        voc = None

    train_loader, val_loader, collate_fn = prepare_dataloaders(conf)

    adversarial_training = conf['train']['adversarial_training']
    if adversarial_training:
        optimizer_g = optimizers['optimizer_g']
        optimizer_d = optimizers['optimizer_d']
        scheduler_g = schedulers['scheduler_g']
        scheduler_d = schedulers['scheduler_d']
    else:
        optimizer = optimizers['optimizer']
        scheduler = optimizers['scheduler']

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
            if adversarial_training:
                loss, d_in_real, d_in_fake = model(step='g',
                                                   batch=batch,
                                                   logger=logger, gs=gs, epoch=epoch,
                                                   dynamic_guide=dynamic_guide)
                optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                loss.mean().backward()
                optimizer_g.step()

                loss, _, _ = model(step='d', batch=batch, d_in_real=d_in_real, d_in_fake=d_in_fake,
                                   logger=logger, gs=gs, epoch=epoch)
                optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                loss.mean().backward()
                optimizer_d.step()

            else:
                loss, _ = model(batch=batch,
                                dynamic_guide=dynamic_guide,
                                logger=logger,
                                gs=gs)

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
            bar.set_description("ep: {}, lr: {}, gs: {}".format(epoch, optimizer_g.state_dict()['param_groups'][0]['lr'], gs))
            gs += 1

        if adversarial_training:
            scheduler_g.step()
            scheduler_d.step()
        else:
            scheduler.step()

        if (epoch + 1) % int(conf['train']['valid_epoch']) == 0:
            valdataiter = iter(val_loader)
            traindataiter = iter(train_loader)
            for i in range(int(conf['train']['valid_num'])):
                val_batch = next(valdataiter)
                train_batch = next(traindataiter)
                validate(conf, model, train_batch, logger, gs, dynamic_guide, validation_name='train', valid_num=i, voc=voc)
                validate(conf, model, val_batch, logger, gs, dynamic_guide, validation_name='valid', valid_num=i, voc=voc)

        if (epoch + 1) % int(conf['train']['save_epoch']) == 0:
            checkpoint_path = os.path.join(
                output_directory, "checkpoint_{}".format(gs))
            if adversarial_training:
                save_checkpoint(model,
                                {"optimizer_g": optimizer_g,
                                 "optimizer_d": optimizer_d},
                                {"scheduler_g": scheduler_g,
                                 "scheduler_d": scheduler_d},
                                float(conf['optimizer']['adam_alpha']),
                                gs, checkpoint_path)

            else:
                save_checkpoint(model,
                                {"optimizer": optimizer},
                                float(conf['optimizer']['adam_alpha']),
                                gs, checkpoint_path)