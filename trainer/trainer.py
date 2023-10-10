import os
import fnmatch
import numpy as np
import sklearn.metrics as skmetrics
import torch

import model.metric as module_metric
from base import BaseTrainer
from utils import inf_loop, MetricTracker, NativeScalerWithGradNormCount
from timm.utils import AverageMeter


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None, cv=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.device = device
        self.cv = cv
        self.show_details = config['trainer']['show_details']
        self.mixed_precision = config['trainer']['mixed_precision']
        self.accumulation_steps = config['trainer']['accumulate_steps']
        self.clip_grad = config['trainer']['clip_grad']
        self.total_epoch = config['trainer']['epochs']
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size)) * self.accumulation_steps

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.loss_scaler = NativeScalerWithGradNormCount()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.train_metrics.reset()
        self.model.train()
        self.optimizer.zero_grad()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()
        outputs_epoch = []
        targets_epoch = []
        iter = 0

        for batch_idx, (idx, data, targets) in enumerate(self.data_loader):
            targets = targets.type(torch.LongTensor)
            data, targets = data.to(self.device), targets.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.float16):
                outputs = self.model(data)
            if iter == 0:
                outputs_epoch = outputs
                targets_epoch = targets
            else:
                outputs_epoch = torch.cat([outputs_epoch, outputs])
                targets_epoch = torch.cat([targets_epoch, targets])

            loss = self.criterion(outputs, targets)
            loss = loss / self.accumulation_steps + 1e-9
            is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
            grad_norm = self.loss_scaler(loss, self.optimizer, clip_grad=self.clip_grad,
                                         parameters=self.model.parameters(), create_graph=is_second_order,
                                         update_grad=(batch_idx + 1) % self.accumulation_steps == 0)

            if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(self.data_loader)):
                self.optimizer.zero_grad()
                self.lr_scheduler.step((epoch * self.len_epoch + batch_idx) // self.accumulation_steps)

            loss_scale_value = self.loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            loss_meter.update(loss.item(), targets.size(0))
            if grad_norm is not None:
                norm_meter.update(grad_norm)
            scaler_meter.update(loss_scale_value)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item()*self.accumulation_steps)

            if batch_idx % self.log_step == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                self.logger.info(
                    f'Train: [{epoch}/{self.total_epoch}][{batch_idx}/{self.len_epoch}]\t'
                    f'Train loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Train grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'Train loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                    f'Train mem {memory_used:.0f}MB')

            if batch_idx == self.len_epoch:
                break
            iter += 1

        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(outputs_epoch, targets_epoch))
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        return log

    @torch.no_grad()
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        loss_meter = AverageMeter()
        pid = []
        truth = []
        logits = []

        for batch_idx, (idx, data, targets) in enumerate(self.valid_data_loader):
            targets = targets.type(torch.LongTensor)
            data, targets = data.to(self.device), targets.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.float16):
                outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            self.valid_metrics.update('loss', loss.item())
            loss_meter.update(loss.item(), targets.size(0))
            pid.append(list(idx))
            truth.append(targets.tolist())
            logits.append(torch.sigmoid(outputs).tolist())
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))

        if self.show_details:
            logits = list(np.squeeze(sum(logits, [])))
            truth = sum(truth, [])
            threshold = module_metric.Find_Optimal_Cutoff(logits, truth)
            pred = list((logits >= threshold) * 1)
            roc_auc = skmetrics.roc_auc_score(truth, logits)
            acc = skmetrics.roc_auc_score(truth, pred)

            self.logger.info(
                             f'Valid Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                             f'Valid AUC: {np.round(roc_auc, 5)}\t'
                             f'Valid Acc: {np.round(acc, 5)}'
                             )
        return self.valid_metrics.result()

    def evaluate(self, mode='valid', verbose=False):
        best_auc = best_acc = best_prec = best_sens = best_spec = 0.
        best_pid = best_cutoff = best_probability = best_predicted = best_target = best_epoch = 0.

        if mode == 'valid':
            eval_loader = self.valid_data_loader
        elif mode == 'test':
            eval_loader = self.test_data_loader
        if isinstance(self.cv, int):
            saved_files = sorted(list(filter(lambda x: 'fold' + str(self.cv) in x, (os.listdir(self.checkpoint_dir)))))
        else:
            saved_files = sorted(list(filter(lambda x: '.pth' in x, (os.listdir(self.checkpoint_dir)))))
        for fname in saved_files:
            if fnmatch.fnmatch(fname, 'model_best.pth'):
                model_path = os.path.join(self.checkpoint_dir, fname)
                checkpoint = torch.load(model_path)
                state_dict = checkpoint['state_dict']
                self.model.load_state_dict(state_dict)

                # prepare model for testing
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                self.model.eval()

                pid = []
                truth = []
                pred = []

                with torch.no_grad():
                    for batch_idx, (idx, data, targets) in enumerate(eval_loader):
                        targets = targets.type(torch.LongTensor)
                        data, targets = data.to(self.device), targets.to(self.device)

                        with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.float16):
                            outputs = self.model(data)
                        pid.append(np.asarray(idx))
                        truth.append(targets.cpu().detach().numpy().ravel())
                        pred.append(torch.sigmoid(outputs).cpu().detach().numpy().ravel())

                preds = np.concatenate(pred, axis=0)
                pred_value = preds.flatten()
                targets = np.concatenate(truth, axis=0)

                roc_auc = skmetrics.roc_auc_score(targets, pred_value)
                threshold = module_metric.Find_Optimal_Cutoff(pred_value, targets)
                accuracy = skmetrics.accuracy_score(targets, (pred_value >= threshold))
                precision = skmetrics.precision_score(targets, (pred_value >= threshold))
                recall = skmetrics.recall_score(targets, (pred_value >= threshold))
                tn, fp, fn, tp = skmetrics.confusion_matrix(targets, (pred_value >= threshold)).ravel()
                neg_recall = tn / (tn + fp)

                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_prec = precision
                    best_sens = recall
                    best_spec = neg_recall
                    best_acc = accuracy

                    best_pid = pid
                    best_cutoff = threshold
                    best_probability = pred_value
                    best_predicted = (pred_value >= threshold) * 1
                    best_target = targets
                    best_epoch = fname[22:25]

                if verbose:
                    print(model_path)
                    print('corresponding ID: ')
                    print(list(pid))
                    print('predicted probability: ')
                    print(list(pred_value))
                    print('predicted value: ')
                    print(list((pred_value >= threshold) * 1))
                    print('truth: ')
                    print(list(targets))

                    print("Model performance in test set: fold %d" % str(self.cv))
                    print("auc: %.4f " % roc_auc)
                    print("accuracy: %.4f " % accuracy)
                    print("precision: %.4f " % precision)
                    print("sensitivity: %.4f " % recall)
                    print("specificity: %.4f " % neg_recall)

        return best_auc, best_prec, best_sens, best_spec, best_acc, \
               best_pid, best_cutoff, best_probability, best_predicted, best_target, best_epoch

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
