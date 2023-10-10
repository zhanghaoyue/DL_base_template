import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
import nibabel as nib
import pandas as pd
import os
import torch
import numpy as np
import random
from scipy import interpolate
from torch import inf


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# for healthy half vs stroke half prediction
def filter_dataframe(params, holdout=False):
    if not os.path.exists(params['files']['patient_file_filter']):
        patient_df = pd.read_csv(params['files']['patient_file']).dropna(axis=0, how='all')
        print(patient_df.shape)
        pts_with_imgs = []
        # filter by label first
        patient_df = patient_df[(patient_df[params['inputs']['label']] != 'na')]
        print(patient_df.shape)
        patient_df = patient_df.dropna()
        print(patient_df.shape)
        patient_df[params['inputs']['label']] = patient_df[params['inputs']['label']].astype(int)

        path1 = []
        path2 = []
        # select series here, but this is very ugly
        if params['inputs']['selection'] == 'first':
            for idx in sorted(list(patient_df['ID'])):
                img_path = os.path.join(params['files']['data_location'], str(idx) + '_IR')
                if os.path.exists(img_path):
                    print(idx)
                    folder = idx + '_IR'
                    series = []
                    for f in os.listdir(img_path):
                        if f.endswith('nii.gz'):
                            if nib.load(os.path.join(img_path, f)).get_fdata().shape[2] > 1:
                                series.append(f)
                    if len(series) > 6:
                        if sorted(series)[2][:3] == sorted(series)[3][:3]:
                            series = sorted(series)[2:4]
                            pts_with_imgs.append(idx)
                        elif sorted(series)[3][:3] == sorted(series)[4][:3]:
                            series = sorted(series)[3:5]
                            pts_with_imgs.append(idx)
                        elif sorted(series)[4][:3] == sorted(series)[5][:3]:
                            series = sorted(series)[4:6]
                            pts_with_imgs.append(idx)
                        else:
                            continue
                    else:
                        continue

                    if idx in pts_with_imgs:
                        print(series)
                        path1.append(series[0])
                        path2.append(series[1])

        elif params['inputs']['selection'] == 'last':
            for idx in sorted(list(patient_df['ID'])):
                img_path = os.path.join(params['files']['data_location'], str(idx) + '_IR')
                if os.path.exists(img_path):
                    print(idx)
                    folder = idx + '_IR'
                    series = []
                    for f in os.listdir(img_path):
                        if f.endswith('nii.gz'):
                            if nib.load(os.path.join(img_path, f)).get_fdata().shape[2] > 1:
                                series.append(f)
                    sort_ser = sorted(series, reverse=True)
                    if len(series) > 6:
                        if sort_ser[2][:3] == sort_ser[3][:3]:
                            series = sort_ser[2:4]
                            pts_with_imgs.append(idx)
                        elif sort_ser[3][:3] == sort_ser[4][:3]:
                            series = sort_ser[3:5]
                            pts_with_imgs.append(idx)
                        elif sort_ser[4][:3] == sort_ser[5][:3]:
                            series = sort_ser[4:6]
                            pts_with_imgs.append(idx)
                        else:
                            continue
                    else:
                        continue

                    if idx in pts_with_imgs:
                        print(series)
                        path1.append(series[0])
                        path2.append(series[1])

        filtered_df = patient_df[patient_df['ID'].isin(pts_with_imgs)]
        filtered_df = filtered_df.sort_values('ID', inplace=False)
        filtered_df['image_path1'] = path1
        filtered_df['image_path2'] = path2
        filtered_df.to_csv(params['files']['patient_file_filter'])
    else:
        filtered_df = pd.read_csv(params['files']['patient_file_filter'])
    patients = filtered_df['ID'].tolist()
    if holdout:
        patients_test = pd.read_csv(params['files']['test_file']).dropna(axis=0, how='all')
        test_index = [str(i) for i in patients_test['ID'].tolist()]
    else:
        test_index = []
    train_index = [x for x in patients if x not in test_index]
    df_train, df_test = filtered_df[filtered_df['ID'].isin(train_index)], filtered_df[
        filtered_df['ID'].isin(test_index)]

    return df_train, df_test


def to_one_hot(x, C=1, tensor_class=torch.cuda.FloatTensor):
    """ One-hot a batched tensor of shape (B, ...) into (B, C, ...) """
    x_one_hot = tensor_class(x.size(0), C, *x.shape[1:]).zero_()
    x_one_hot = x_one_hot.scatter_(1, x.unsqueeze(1), 1)
    return x_one_hot


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def load_pretrained(path, model):
    print(f">>>>>>>>>> Fine-tuned from pretrained model ..........")
    checkpoint = torch.load(path, map_location='cpu')
    try:
        checkpoint_model = checkpoint['model']
    except KeyError:
        checkpoint_model = checkpoint['state_dict']

    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if
                            k.startswith('encoder.')}
        print('Detect pre-trained model, remove [encoder.] prefix.')
    else:
        print('Detect non-pre-trained model, pass without doing anything.')

    print(f">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
    checkpoint = remap_pretrained_keys_swin(model, checkpoint_model)

    del checkpoint
    torch.cuda.empty_cache()
    print(f">>>>>>>>>>pretrained weight loaded successfully")


def remap_pretrained_keys_swin(model, checkpoint_model):
    state_dict = model.state_dict()

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            import pdb
            pdb.set_trace()
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    print(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _init_fn(worker_id, SEED):
    seed_torch(SEED)
