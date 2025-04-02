from typing import Any

import numpy as np
import torch
import torch.nn as nn
from .base_postprocessor import BasePostprocessor
import numpy as np
from .info import num_classes_dict

class POPPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False
        self.num_classes = num_classes_dict[self.config.dataset.name]

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output,  feature_ood = net(data, return_feature=True)
        feature_norm = torch.norm(feature_ood, dim=-1)
        maxlogit, pred = torch.max(output, dim=1)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        score_ood = (feature_norm * maxlogit).cpu().numpy() 

        return pred, torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim