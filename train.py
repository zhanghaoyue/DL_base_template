import argparse
import collections
import warnings
import random
import numpy as np
import os
import torch

import data_loader.dataset_sample as dsa_loader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_pool
from base import base_data_loader
from parse_config import ConfigParser
from sklearn.model_selection import train_test_split
from trainer import Trainer
from utils import util
import timm

warnings.filterwarnings("ignore")


def main(pars, opts):
    config = ConfigParser.from_args(pars, opts)
    params = config.config
    util.seed_torch(pars.parse_args().seed)
    torch.autograd.set_detect_anomaly(True)

    logger = config.get_logger('train')

    df_train, _ = util.filter_dataframe(params)
    labels = np.array(df_train[params['inputs']['label']])

    df_dev, df_valid, _, _ = train_test_split(df_train, labels, test_size=params['inputs']['test_size'])

    # setup data_loader instances
    X_train = dsa_loader.DSA_Dataset(df_dev, params)
    X_valid = dsa_loader.DSA_Dataset(df_valid, params)

    train_loader_par = params['data_loader_train']['args']
    train_loader = base_data_loader.BaseDataLoader(X_train,
                                                   batch_size=train_loader_par['batch_size'],
                                                   shuffle=train_loader_par['shuffle'],
                                                   num_workers=train_loader_par['num_workers'],
                                                   validation_split=train_loader_par['validation_split'])

    valid_loader_par = params['data_loader_valid']['args']
    valid_loader = base_data_loader.BaseDataLoader(X_valid,
                                                   batch_size=valid_loader_par['batch_size'],
                                                   shuffle=valid_loader_par['shuffle'],
                                                   num_workers=valid_loader_par['num_workers'],
                                                   validation_split=valid_loader_par['validation_split'])

    # build model architecture, then print to console
    model = config.init_obj('arch', module_pool)
    # logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = util.prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_loader,
                      valid_data_loader=valid_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Stroke', conflict_handler="resolve")
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default="0", type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-s', '--seed', default='2023', type=int,
                        help='seed number (default: 2023)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    main(parser, options)
