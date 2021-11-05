import torch, math, time, argparse, os
import random, dataset, utils, losses, net
import numpy as np

from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from tqdm import *
import wandb

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # set random seed for all gpus

parser = argparse.ArgumentParser(description=
                                 'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'
                                 + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`'
                                 )
# export directory, training and val datasets, test datasets

parser.add_argument('--workers', default=4, type=int, dest='nb_workers', help='Number of workers for dataloader.')
parser.add_argument('--l2-norm', default=1, type=int, help='L2 normlization')
parser.add_argument('--bn-freeze', default = 1, type = int,help = 'Batch normalization parameter freeze')
parser.add_argument('--remark', default='', help='Any reamrk')

parser.add_argument('--LOG_DIR', default='../logs/cars_512', help='Path to log folder')
parser.add_argument('--batch-size', default=32, type=int, dest='sz_batch', help='Number of samples per batch.')
parser.add_argument('--dataset', default='cars', help='Training dataset, e.g. cub, cars, SOP, Inshop, logo2k')
parser.add_argument('--data_root', default='/home/ruofan/PycharmProjects/ProxyNCA-/mnt/datasets/')
parser.add_argument('--embedding-size', default=512, type=int, dest='sz_embedding',
                    help='Size of embedding that is appended to backbone model.')
parser.add_argument('--model', default='resnet50', help='Model for training')
parser.add_argument('--loss', default='MS', help='Criterion for training')
parser.add_argument('--gpu-id', default=1, type=int, help='ID of GPU that is used for training. -1 means use all')
parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate setting')

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Directory for Log
LOG_DIR = args.LOG_DIR
data_root = args.data_root

if args.dataset != 'Inshop':
    ev_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode='eval',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

else:
    query_dataset = Inshop_Dataset(
        root=data_root,
        mode='query',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

    gallery_dataset = Inshop_Dataset(
        root=data_root,
        mode='gallery',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )


# Backbone Model
if args.model.find('googlenet') + 1:
    model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
elif args.model.find('bn_inception') + 1:
    model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                         bn_freeze=args.bn_freeze)
elif args.model.find('resnet18') + 1:
    model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
elif args.model.find('resnet50') + 1:
    model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
elif args.model.find('resnet101') + 1:
    model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
model = model.cuda()
model.load_state_dict(torch.load('{}/{}_{}_{}_best.pth'.format(LOG_DIR, args.dataset, args.model, args.loss))['model_state_dict'])

print("**Evaluating...**")
if args.dataset == 'Inshop':
    utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
elif args.dataset != 'SOP':
    utils.evaluate_cos(model, dl_ev)

