import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torchvision.models.resnet import BasicBlock, ResNet
from .HAFrame.solve_HAF import distance_matrix_to_haf_cls_weights
# from haf.arch import HAF_resnet
from ..resnet18_224x224 import ResNet18_224x224
from ..resnet18_32x32 import ResNet18_32x32

from torch.nn.functional import linear, normalize


class ResNet18_fix(nn.Module):
    def __init__(self, num_classes, pooling=max, model=ResNet18_224x224(num_classes=100), haf_cls_weights=None):
        super(ResNet18_fix, self).__init__()
        self.num_classes = num_classes
        
        # cifar 10
        if self.num_classes == 10:
            self.num_others = 10
            self.num_total = self.num_classes + self.num_others
            model = ResNet18_32x32(num_classes=self.num_total)
        # cifar 100
        elif self.num_classes == 100:
            self.num_others = 0 #40
            self.num_total = self.num_classes + self.num_others
            model = ResNet18_32x32(num_classes=self.num_total)
        elif self.num_classes == 3:
            self.num_others = 0 
            self.num_total = self.num_classes + self.num_others
            model = ResNet18_32x32(num_classes=self.num_total)
        else:
            self.num_others = 0
            self.num_total = self.num_classes + self.num_others
            model = ResNet18_224x224(num_classes=self.num_total)
            
        self.num_ftrs = 512 * 1 * 1       # Used for resnet18
        self.haf_gamma = 512
        self.model = model
        # backbone = list(self.model.children())[:-2]
        # add 1x1 conv layer: channel-wise downsampling
        self.conv1 = nn.Conv2d(self.num_ftrs, self.num_total,
                        kernel_size=1, stride=1, padding=0, bias=False)
        # self.features_2 = nn.Sequential(*backbone)
        self.model = model

        if pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=7, stride=7) # pooling
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
                    
        self.classifier_3 = nn.Linear(self.num_total, self.num_total, bias=True)


        if self.num_classes == 10:           # cifar10 distance matrix path
            self.distance_path = "openood/networks/haf/data/cifar10dist.npy"
        elif self.num_classes == 100:        # cifar100 distance matrix path
            self.distance_path = "openood/networks/haf/data/cifar100dist.npy"
        else:                                # img200 distance matrix path
            self.distance_path = "openood/networks/haf/data/imagenet200dist.npy"

        self.temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
        self.haf_cls_weights = self.get_distance(self.distance_path)
        # self.haf_cls_weights = np.vstack((self.haf_cls_weights, np.eye(self.num_classes)))
        if self.haf_cls_weights is not None:
            with torch.no_grad():
                self.classifier_3.weight = nn.Parameter(torch.Tensor(self.haf_cls_weights))
                self.classifier_3.weight.requires_grad_(False)
                self.classifier_3.bias = nn.Parameter(torch.zeros([self.num_total, ]))
                self.classifier_3.bias.requires_grad_(False)

    def forward(self, x, target="ignored", return_feature1=False, return_feature2=False, return_feature_list=False, return_logit=False):
        # x1 = self.features_2(x)
        _, feature_list = self.model(x, return_feature_list=True)
        x2 = self.conv1(feature_list[-1])
        feature_512 = self.pool(feature_list[-1])
        x3 = self.pool(x2)

        # feature_512 = normalize(feature_512)

        feature_10 = x3.view(x3.size(0), -1)

        norm_embeddings = F.normalize(feature_10, p=2, dim=-1)
        norm_weight_activated = normalize(self.classifier_3.weight)
        logit = linear(norm_embeddings, norm_weight_activated)

        # logit = self.classifier_3(feature_10)
        # logit_prev = logit[:, :10]
        # logit_next = logit[:, 10:]
        # logit = logit_prev * torch.softmax(torch.relu(logit_next), dim=1)

        # logit = self.classifier_3(feature_10)

        if return_feature1:
            return logit, feature_512.view(feature_512.size(0), -1)
        elif return_feature2:
            return logit, feature_10
        elif return_feature_list:
            return logit, feature_list
        else:
            return logit

    def get_penultimate_feature(self, x):
        x = nn.Sequential(*list(self.features_2.children())[:-1])(x)
        x = self.pool(x)
        feature = x.view(x.size(0), -1)
        return feature

    def get_fc(self):      # get the classifier weights
        cls = self.classifier_3
        return cls.weight.cpu().detach().numpy(), cls.bias.cpu().detach().numpy()

    def get_distance(self, distance_path, class_str_labels=None):
        distance_matrix = np.load(distance_path)
        
        if self.num_classes == 10:
            with_others = np.zeros((self.num_total, self.num_total))
            
            for i in range(self.num_total):
                for j in range(self.num_total):
                    if i < self.num_classes and j < self.num_classes:
                        with_others[i, j] = distance_matrix[i, j]
                    else:
                        if i == j:
                            with_others[i, j] = 0
                        # elif i < self.num_classes + self.num_others / 2 and j < self.num_classes + self.num_others / 2: #near
                        #     with_others[i, j] = 6
                        else: #far
                            with_others[i, j] = 6
            
            distance_matrix = with_others
            
        elif self.num_classes == 100:
            with_others = np.zeros((self.num_total, self.num_total))
            
            for i in range(self.num_total):
                for j in range(self.num_total):
                    if i < self.num_classes and j < self.num_classes:
                        with_others[i, j] = distance_matrix[i, j]
                    else:
                        if i == j:
                            with_others[i, j] = 0
                        elif i < self.num_classes + self.num_others / 2 and j < self.num_classes + self.num_others / 2: #near
                            with_others[i, j] = 8
                        else: #far
                            with_others[i, j] = 15
            
            distance_matrix = with_others
        
        else:
            with_others = np.zeros((self.num_total, self.num_total))

            for i in range(self.num_total):
                for j in range(self.num_total):
                    if i < self.num_classes and j < self.num_classes:
                        with_others[i, j] = distance_matrix[i, j]
                    else:
                        if i == j:
                            with_others[i, j] = 0
                        elif i < self.num_classes + self.num_others / 2 and j < self.num_classes + self.num_others / 2: #near
                            with_others[i, j] = 18
                        else: #far
                            #with_others[i, j] = 3 * np.max(distance_matrix)#原来的
                            with_others[i, j] = 4 * np.max(distance_matrix)            
            
            distance_matrix = with_others


        # np.random.seed(42)
        # distance_matrix = np.random.rand(self.num_classes, self.num_classes)
        # distance_matrix = np.load(distance_path)
        haf_cls_weights, _, _, mapping_function = \
            distance_matrix_to_haf_cls_weights(distance_matrix,
                                                class_str_labels,
                                                self.num_total,
                                                self.haf_gamma)
        # self.mapping_function = mapping_function
        # eigenvalues, eigenvectors = np.linalg.eigh(distance_matrix)
        # sqrt_eigenvalues = np.diag(np.abs(eigenvalues))
        # haf_cls_weights = np.dot(eigenvectors, sqrt_eigenvalues)

        return haf_cls_weights



    
