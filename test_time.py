import gc
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import cfg
from models.CC import CrowdCounter


def measure(model, x):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    return time.time() - t0


def benchmark(model, x, n_warmup=10, n_iters=10):
    model = model.cuda().eval()
    for _ in range(n_warmup):
        _ = measure(model, x)
    print('DONE WITH DRY RUNS, NOW BENCHMARKING')
    t_forward = [measure(model, x) for _ in range(n_iters)]
    del model
    return t_forward


def main():
    torch.manual_seed(1234)
    cudnn.benchmark = True

    model = CrowdCounter(cfg.GPU_ID, cfg.NET).CCN
    print('Benchmarking {} on 1920x1080 input'.format(cfg.NET))

    batch_sizes = [1, 2, 4]
    mean_tfp, std_tfp = [], []
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 1920, 1080).cuda()
        tmp = benchmark(model, x)
        mean_tfp.append(np.asarray(tmp).mean() / bs * 1e3)
        std_tfp.append(np.asarray(tmp).std() / bs * 1e3)
        print('batch={}: {:.2f} ms/image'.format(bs, mean_tfp[-1]))

    print('mean ms/image:', mean_tfp)
    print('std ms/image: ', std_tfp)

    gc.collect()


if __name__ == '__main__':
    main()
