import os, sys, sklearn
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.decomposition.pca as PCA
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
import seaborn as sns
sns.set()
from sklearn.metrics import r2_score
import warnings
warnings.simplefilter('ignore')
from sklearn import linear_model
from umap import UMAP
import copy
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string
from matplotlib.colors import ListedColormap
from numba import jit
import scanpy.api as sc
import anndata
import scvelo as scv
script_path = (os.path.dirname(os.path.realpath(__file__)))
sys.path.append(script_path)
sys.path.append(script_path+'/utils/')
input_folder_dir = os.path.join(script_path, '../../Submission_analysis/Data/')
from utils import sc_tools as sat
