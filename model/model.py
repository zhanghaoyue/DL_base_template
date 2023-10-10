import torchvision

from model.arch.deeptici import TICIModelHandler as tici
# from model.arch.mtv import MTVMultiheadClassificationModel as mtv
from model.arch.spatiotemporaltransformer import Net
from model.arch.timesformer import TimeSformer as tsformer
import model.arch.swin_transformer_3d as swin3d
import utils
from collections import OrderedDict
import torch
import pathlib


def TICIModelHandler(pretrained=False, **kwargs):
    model = tici(num_classes=kwargs['num_classes'], feature_size=kwargs['feature_size'])
    return model


def STNet(pretrained=False, **kwargs):
    model = Net()
    return model


def swin3D(num_cls=1, **kwargs):
    model = swin3d.SwinTransformer(
        img_size=(32, 224, 224),
        patch_size=(4, 4, 4),
        in_chans=2,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_classes=num_cls,
        num_heads=[3, 6, 12, 24],
        window_size=(4, 7, 7),
        mlp_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        use_checkpoint=True)

    if kwargs['pretrained']:
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        kwargs['init_weights'] = False
        path = kwargs['model_path']
        if "window" in path:
            utils.load_pretrained(path, model)
        elif "lvo" in path:
            new_state_dict = OrderedDict()

            kwargs['init_weights'] = False
            checkpoint = torch.load(path)
            state_dict = checkpoint['state_dict']
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)
            print("pretrained model loaded successfully! <============>")
            print("pretrained model name: %s" % kwargs['model_path'])

    return model


def TimeSformer(**kwargs):
    model = tsformer()
    return model


def resnet3d(num_cls=1, **kwargs):
    class r3d_18(torch.nn.Module):
        def __init__(self, base):
            super(r3d_18, self).__init__()
            self.base_model = base

        def forward(self, x):
            x = torch.stack([x[:, 0, :, :, :], x[:, 0, :, :, :], x[:, 0, :, :, :]], 1)
            return self.base_model(x)

    base_model = torchvision.models.video.r3d_18(pretrained=True)
    base_model.fc = torch.nn.Linear(512, num_cls)
    model = r3d_18(base_model)
    if kwargs['pretrained']:

        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

        new_state_dict = OrderedDict()

        kwargs['init_weights'] = False
        checkpoint = torch.load(kwargs['model_path'])
        state_dict = checkpoint['state_dict']
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        print("pretrained model loaded successfully! <============>")
        print("pretrained model name: %s" % kwargs['model_path'])
    else:
        kwargs['init_weights'] = True

    return model