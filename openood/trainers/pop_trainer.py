import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import openood.utils.comm as comm
from openood.utils import Config
from .lr_scheduler import cosine_annealing
import torch
from torch.autograd import Function
from torch.nn.functional import linear, normalize

class POPTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.s = self.config['optimizer']['s']
        self.num_classes = self.config.dataset.num_classes
        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

        self.loss_hsbl = HSBL(self.net, self.s)


    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            logits, feature = self.net(data, return_feature=True) 

            loss = self.loss_hsbl(logits, target, self.s)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced

class HSBL(nn.Module):
    def __init__(self, net, s):
        super().__init__()
        self.net = net
        self.fix_layer = net.classifier_3.weight
        self.gamma = 1
        self.s = s
    def hsbl_loss(self, logits: torch.Tensor, labels: torch.Tensor, temp):
        cls_weight = torch.tensor(self.net.get_cls(), dtype=torch.float32, device=logits.device)
        pred = torch.argmax(logits, dim=-1)
        m = 1 - torch.einsum('ij,ij->i', cls_weight[pred], cls_weight[labels])[:, None]
        cos_theta = logits
        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.bool()
        cos_theta_m = cos_theta - m
        output = temp * cos_theta_m
        loss = F.cross_entropy(output, labels)
        return loss

    def forward(self, y_pred, y_true, temp):
        return self.hsbl_loss(y_pred, y_true, temp)