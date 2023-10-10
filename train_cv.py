import argparse
import collections
import gc
import warnings
import copy
import numpy as np
import scipy.stats as st
import torch
from sklearn.model_selection import StratifiedKFold

import model.loss as module_loss
import model.metric as module_metric
import model.model as module_pool
from data_loader import data_loaders as dl
from parse_config import ConfigParser
from trainer import Trainer
from utils import util, prepare_device

warnings.filterwarnings("ignore")


def main(pars, opts):
    # Fix random seeds for reproducibility
    config = ConfigParser.from_args(pars, opts)
    params = config.config
    SEED = pars.parse_args().seed
    util.seed_torch(SEED)
    torch.autograd.set_detect_anomaly(True)
    print("seed using is:" + str(SEED))

    # Filter and split up data
    df_dev, _ = util.filter_dataframe(params, holdout=False)
    labels = np.array(df_dev[params['inputs']['label']])
    print("Total cases: ")
    print(len(df_dev['ID'].tolist()))
    print("positive cases: ")
    print(sum(labels == 1))
    print("negative cases: ")
    print(sum(labels == 0))

    # Split into cross-validated folds
    skf = StratifiedKFold(n_splits=params['trainer']['cv_fold'], shuffle=True, random_state=SEED)
    skf.get_n_splits(df_dev, labels)

    # Declare dictionaries of performance metrics
    best_auc = [0.] * int(params['trainer']['cv_fold'])
    best_acc = [0.] * int(params['trainer']['cv_fold'])
    best_prec = [0.] * int(params['trainer']['cv_fold'])
    best_sens = [0.] * int(params['trainer']['cv_fold'])
    best_spec = [0.] * int(params['trainer']['cv_fold'])

    best_pid = [0.] * int(params['trainer']['cv_fold'])
    best_cutoff = [0.] * int(params['trainer']['cv_fold'])
    best_probability = [0.] * int(params['trainer']['cv_fold'])
    best_predicted = [0.] * int(params['trainer']['cv_fold'])
    best_target = [0.] * int(params['trainer']['cv_fold'])
    best_epoch = [0.] * int(params['trainer']['cv_fold'])

    i = 1
    for train_index, valid_index in skf.split(df_dev, labels):

        df_train, df_valid = df_dev.iloc[train_index], df_dev.iloc[valid_index]

        # Instantiate all data loaders
        train_loader = dl.dsa_pair_pre(df_train, params, type='train')
        valid_loader = dl.dsa_pair_pre(df_valid, params, type='valid')

        # build model architecture, then print to console
        model = config.init_obj('arch', module_pool)
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # Build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        init_state = copy.deepcopy(model.state_dict())
        init_state_opt = copy.deepcopy(optimizer.state_dict())
        model.load_state_dict(init_state)
        optimizer.load_state_dict(init_state_opt)

        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        # Initialize trainer
        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=train_loader,
                          valid_data_loader=valid_loader,
                          lr_scheduler=lr_scheduler,
                          cv=i
                          )

        trainer.train()

        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate the model
        best_auc[i - 1], best_prec[i - 1], best_sens[i - 1], best_spec[i - 1], best_acc[i - 1], \
        best_pid[i - 1], best_cutoff[i - 1], best_probability[i - 1], best_predicted[i - 1], best_target[i - 1], \
        best_epoch[i - 1] = trainer.evaluate()

        i += 1

    print("Patient IDs:" + str(best_pid))
    print("Best Cutoffs:" + str(best_cutoff))
    print("Probabilities:" + str(best_probability))
    print("Predicted Values:" + str(best_predicted))
    print("Target Values:" + str(best_target))
    print("Best Epochs:" + str(best_epoch))
    print('')
    print("ROC-AUC:     %.4f (%.4f)" % (np.mean(np.array(best_auc)), np.std(np.array(best_auc))))
    print("Accuracy:    %.4f (%.4f)" % (np.mean(np.array(best_acc)), np.std(np.array(best_acc))))
    print("Sensitivity: %.4f (%.4f)" % (np.mean(np.array(best_sens)), np.std(np.array(best_sens))))
    print("Specificity: %.4f (%.4f)" % (np.mean(np.array(best_spec)), np.std(np.array(best_spec))))
    print("Precision:   %.4f (%.4f)" % (np.mean(np.array(best_prec)), np.std(np.array(best_prec))))

    print('')
    range_auc = st.t.interval(confidence=0.95, df=len(best_auc) - 1, loc=np.mean(best_auc), scale=st.sem(best_auc))
    print("AUC:  %.4f (%.4f-%.4f)" % (np.mean(np.array(best_auc)), range_auc[0], range_auc[1]))

    range_acc = st.t.interval(confidence=0.95, df=len(best_acc) - 1, loc=np.mean(best_acc), scale=st.sem(best_acc))
    print("ACC:  %.4f (%.4f-%.4f)" % (np.mean(np.array(best_acc)), range_acc[0], range_acc[1]))

    range_sen = st.t.interval(confidence=0.95, df=len(best_sens) - 1, loc=np.mean(best_sens), scale=st.sem(best_sens))
    print("SENS: %.4f (%.4f-%.4f)" % (np.mean(np.array(best_sens)), range_sen[0], range_sen[1]))

    range_spe = st.t.interval(confidence=0.95, df=len(best_spec) - 1, loc=np.mean(best_spec), scale=st.sem(best_spec))
    print("SPEC: %.4f (%.4f-%.4f)" % (np.mean(np.array(best_spec)), range_spe[0], range_spe[1]))

    range_pre = st.t.interval(confidence=0.95, df=len(best_prec) - 1, loc=np.mean(best_prec), scale=st.sem(best_prec))
    print("PREC: %.4f (%.4f-%.4f)" % (np.mean(np.array(best_prec)), range_pre[0], range_pre[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Stroke', conflict_handler="resolve")
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--seed', default='2023', type=int,
                        help='seed number (default: 2021)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    main(parser, options)
