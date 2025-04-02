import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torchvision.models.resnet import BasicBlock, ResNet
from .HAFrame.solve_HAF import distance_matrix_to_haf_cls_weights
from .better_mistakes.trees import load_distances
# from haf.arch import HAF_resnet
from ..resnet18_224x224 import ResNet18_224x224
from ..resnet50 import ResNet50
from torch.nn.functional import linear, normalize


class ResNet50_fix(nn.Module):
    def __init__(self, num_classes, pooling=max, model=ResNet18_224x224(num_classes=100), haf_cls_weights=None):
        super(ResNet50_fix, self).__init__()
        self.num_classes = num_classes
        
        self.num_others = 0
        self.num_total = self.num_classes + self.num_others
        model = ResNet50(num_classes=self.num_classes)
        pre_train = torch.load("/home/gmr/ood/OpenOOD/weights/resnet50_imagenet1k_v1.pth")
        model.load_state_dict(pre_train, strict=True)    
        self.num_ftrs = 2048 * 1 * 1       # Used for resnet50
        self.haf_gamma = 3
        self.model = model
        backbone = list(self.model.children())[:-2]
        # add 1x1 conv layer: channel-wise downsampling

        backbone.append(nn.Conv2d(self.num_ftrs, self.num_total,
                        kernel_size=1, stride=1, padding=0, bias=False))
        self.features_2 = nn.Sequential(*backbone)

        if pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=7, stride=7) # pooling
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_3 = nn.Linear(self.num_total, self.num_total, bias=True)
        # img200 distance matrix path
        self.distance_path = "openood/networks/haf/data/imagenetdist.npy"
        
        self.haf_cls_weights = self.get_distance(self.distance_path)

        if self.haf_cls_weights is not None:
            with torch.no_grad():
                self.classifier_3.weight = nn.Parameter(torch.Tensor(self.haf_cls_weights))
                self.classifier_3.weight.requires_grad_(False)
                self.classifier_3.bias = nn.Parameter(torch.zeros([self.num_total, ]))
                self.classifier_3.bias.requires_grad_(False)

    def forward(self, x, target="ignored", return_feature=False):
        x1 = self.features_2(x)
        x2 = self.pool(x1)
        feature2 = x2.view(x2.size(0), -1)
        # feature2 = feature2 / np.linalg.norm()
        norm_embeddings = normalize(feature2)
        norm_weight_activated = normalize(self.classifier_3.weight)
        logit = linear(norm_embeddings, norm_weight_activated)
        # logit = self.classifier_3(feature2) #
    

        feature1 = self.get_penultimate_feature(x)

        if return_feature:
            return logit, feature1
        else:
            return logit

    def get_penultimate_feature(self, x):
        x = nn.Sequential(*list(self.features_2.children())[:-1])(x)
        x = self.pool(x)
        feature = x.view(x.size(0), -1)
        return feature

    def get_cls(self):      # get the classifier weights
        cls = self.classifier_3
        return cls.weight.cpu().detach().numpy()

    def get_distance(self, distance_path, class_str_labels=None):

        distance_matrix = np.load(distance_path)
        
        # with_others = np.zeros((self.num_total, self.num_total))
        
        # for i in range(self.num_total):
        #     for j in range(self.num_total):
        #         if i < self.num_classes and j < self.num_classes:
        #             with_others[i, j] = distance_matrix[i, j]
        #         else:
        #             if i == j:
        #                 with_others[i, j] = 0
        #             elif i < self.num_classes + self.num_others / 2 and j < self.num_classes + self.num_others / 2: #near
        #                 with_others[i, j] = 18  # 4 5 6 7 8 9 10
        #             else: #far
        #                 with_others[i, j] = 80
        
        # distance_matrix = with_others
        
        self.distance_matrix = distance_matrix
        haf_cls_weights, _, _, mapping_function = \
            distance_matrix_to_haf_cls_weights(distance_matrix,
                                                class_str_labels,
                                                self.num_total,
                                                self.haf_gamma)
        self.mapping_function = mapping_function
        return haf_cls_weights
