import argparse
import collections
import warnings

import numpy as np
import torch

import model.loss as module_loss
import model.metric as module_metric
import model.model as module_pool
from data_loader import data_loaders as dl
from parse_config import ConfigParser
from utils import util

warnings.filterwarnings("ignore")


def main(pars, opts):
    # Fix random seeds for reproducibility
    config = ConfigParser.from_args(pars, opts)
    params = config.config
    SEED = pars.parse_args().seed
    util.seed_torch(pars.parse_args().seed)
    torch.autograd.set_detect_anomaly(True)

    # Filter and split up data
    df_dev, _ = util.filter_dataframe(params, holdout=False)
    labels = np.array(df_dev[params['inputs']['label']])

    train_loader = dl.dsa_pair_pre(df_dev, params, type='train')


    model = config.init_obj('arch', module_pool)

    # Prepare for (multi-device) GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # Build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    for batch_idx, (idx, data, targets) in enumerate(train_loader):
        targets = targets.type(torch.LongTensor)
        data, targets = data.to(device), targets.to(device)

        outputs = model(data)
        print(idx[0])
        print(data.shape)
        print(outputs)
        # if torch.isnan(outputs).any():
        #     print(idx[0])
        #     print(data.shape)
        #     print(torch.min(data))
        #     print(torch.max(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Stroke', conflict_handler="resolve")
    parser.add_argument('-c', '--config', default='dgx_videoswin.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--seed', default='2021', type=int,
                        help='seed number (default: 2021)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    main(parser, options)
