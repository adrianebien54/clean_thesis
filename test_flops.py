import os
import torch

from config import cfg
from models.CC import CrowdCounter
from models.ptflops import get_model_complexity_info


torch.cuda.set_device(cfg.GPU_ID[0])
torch.backends.cudnn.benchmark = True


def main():
    net = CrowdCounter(cfg.GPU_ID, cfg.NET).CCN
    flops, params = get_model_complexity_info(
        net, (1920, 1080), as_strings=False, print_per_layer_stat=False
    )
    print('Model: {}'.format(cfg.NET))
    print('FLOPs: {}'.format(flops))
    print('Params: {}'.format(params))


if __name__ == '__main__':
    main()
