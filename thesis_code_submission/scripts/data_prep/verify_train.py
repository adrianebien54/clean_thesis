"""Smoke-test the training loop end-to-end against a random in-memory batch.

This bypasses the dataset loader so it can run without any prepared dataset on
disk. It instantiates the model through the same CrowdCounter wrapper that
trainer.py uses, then performs one forward + backward + optimizer step.

Usage:
    python scripts/data_prep/verify_train.py [NET_NAME]
        NET_NAME defaults to MobileCount; any name registered in models/CC.py
        works (MobileCount, MobileCountx1_25, MobileCountx2, CSRNet, MCNN, etc.).
"""
from pathlib import Path
import os
import sys

import torch
import torch.optim as optim

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))

import config
from models.CC import CrowdCounter


def main():
    net_name = sys.argv[1] if len(sys.argv) > 1 else 'MobileCount'
    config.cfg.NET = net_name
    config.cfg.GPU_ID = [0]

    print('Building {}'.format(net_name))
    net = CrowdCounter(config.cfg.GPU_ID, net_name)
    optimizer = optim.Adam(net.CCN.parameters(), lr=config.cfg.LR)

    img = torch.randn(2, 3, 256, 256).cuda()
    gt_map = torch.randn(2, 256, 256).cuda() * 0.01

    net.train()
    optimizer.zero_grad()
    _ = net(img, gt_map)
    loss = net.loss
    loss.backward()
    optimizer.step()

    print('Loss: {:.6f}'.format(loss.item()))
    assert torch.isfinite(loss), 'Loss is not finite'
    print('OK: one optimizer step completed against {}'.format(net_name))


if __name__ == '__main__':
    main()
