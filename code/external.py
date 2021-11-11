#====================================================================
# EXTERNAL MSI-segmentation SETUP
#====================================================================

#====================================================================
# Library import
#====================================================================
import os
import shutil
import time
from datetime import datetime
from pytz import timezone
import math
import random
from argparse import Namespace
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
from PIL import Image
from scipy import ndimage

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
from torch.multiprocessing import cpu_count
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from torchsummary import summary

import torchvision.transforms.functional as tvf
from torchvision import transforms

from sklearn.utils import shuffle as shuffle
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering

# pretrained models
from efficientnet_pytorch import EfficientNet

# import faiss (faiss for nearest neighbor mining in evaluation)
#====================================================================