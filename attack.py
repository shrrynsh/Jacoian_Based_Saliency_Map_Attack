import argparse
import os 
import time
import json 
import numpy as np 
import torch
from torch.utils.data import DataLoader,Subset
from torchvision import dataset,transforms
from tqdm import tqdm

from model import LenNet5, load_model
from jsma import JSMAAttack

def get_args():
    p = argparse.ArgumentParser(description="Run JSMA attack on MNIST")
    p.add_argument("--model_path",     type=str,   default="./checkpoints/lenet_mnist.pth")
    p.add_argument("--data_dir",       type=str,   default="./data")
    p.add_argument("--save_dir",       type=str,   default="./results")
    p.add_argument("--n_samples",      type=int,   default=100,
                   help="Number of source samples to attack")
    p.add_argument("--max_distortion", type=float, default=0.145,
                   help="Max distortion Υ (paper: 0.145 = 14.5%%)")
    p.add_argument("--theta",          type=float, default=1.0,
                   help="Pixel intensity change per iteration (paper: +1)")
    p.add_argument("--strategy",       type=str,   default="increase",
                   choices=["increase", "decrease"],
                   help="Saliency map strategy")
    p.add_argument("--source_class",   type=int,   default=None,
                   help="Attack only this source class (default: all)")
    p.add_argument("--target_class",   type=int,   default=None,
                   help="Attack only this target class (default: all)")
    p.add_argument("--device",         type=str,   default=None)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--verbose",        action="store_true")
    return p.parse_args()



def load_test_data(data_dir: str, n_samples : int, source_class: int=None,seed: int=42):
    transform=transofrms.Compose([transforms.toTensor()])
    dataset=datsets.MNIST(root=data_dir,train=False,download=True,transform=transform)

    if source_class is not None:
        







