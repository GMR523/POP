import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from openood.evaluation_api import Evaluator
from openood.networks.pop import ResNet18_pop
from openood.utils import setup_config

exp = 'cifar100'
config = setup_config()

if exp == 'cifar10':    # dim = 10
    net = ResNet18_pop(num_classes=10, c=2, d=4)
    # net = ResNet18_32x32(num_classes=10)
    path = "/home/gmr/AAAI/OpenOOD/results/AAAI/cifar10_resnet18_pop_pop_e100_lr0.1_alpha3_default/best.ckpt"
elif exp == 'cifar100':
    net = ResNet18_pop(num_classes=100, c=60, d=7)
    path = "/home/gmr/AAAI/OpenOOD/results/AAAI/cifar100_resnet18_pop_pop_e100_lr0.1_alpha3_default/best.ckpt"
else:
    net = Classifier(256, 2048, output_dim=200, args=config.network)
    path = "/home/gmr/OpenOOD/results/szu/cvpr/img200/imagenet200_slot_slot_e100_lr0.1_alpha3_default/best_epoch87_acc0.8330.ckpt"


net.load_state_dict(torch.load(path, map_location='cuda:0'))
net.cuda()
net.eval()

postprocessor_name = "pop"

evaluator = Evaluator(
    net,
    id_name=exp,
    data_root="/home/gmr/OpenOOD/data/",
    config_root="configs/",
    preprocessor=None,                     # default preprocessing for the target ID dataset
    postprocessor_name=postprocessor_name, # the postprocessor to use
    postprocessor=None,                    # if you want to use your own postprocessor
    batch_size=256,                        # for certain methods the results can be slightly affected by batch size
    shuffle=False,
    num_workers=16)

metrics = evaluator.eval_ood(fsood=False)
