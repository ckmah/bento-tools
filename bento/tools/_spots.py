import bento
import numpy as np
import pandas as pd
import pickle
import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from joblib import Parallel, delayed
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint
from tqdm.auto import tqdm

import optuna

from .._settings import settings

def detect_spots(data, imagedir, device='auto', copy=False):
    """
    Detect and label localization patterns.
    TODO change data to be iterable compatible with skorch.predict_proba

    Parameters
    ----------
    data : [type]
        [description]
    imagedir : str
        Folder for bento rasterized images.
    Returns
    -------
    [type]
        [description]
    """
    adata = data.copy() if copy else data

    # Class names
    classes = ['cell2D', 'cellext', 'foci', 'nuc2D', 'polarized', 'random']

    # Default to gpu if possible. Otherwise respect specified parameter
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_dir = '/'.join(bento.__file__.split('/')[:-1]) + '/models/spots'

    # Load temperature values as list
    # with open(f'{model_dir}/temperature.txt', 'r') as f:
    #     temperatures = f.read().splitlines()
    study = optuna.load_study(
        study_name=f"spots_20210117",
        storage=f"sqlite:///{model_dir}/optuna.db",
    )

    # Load ensemble model
    binary_clfs = []
    for i, c in enumerate(classes):


            # Reload model with temperature and softmax
        net = NeuralNetClassifier(
            module=SpotsModule,
            module__params=study.best_params,
            module__eval_prob=True,
            # module__temperature=temperatures[i]
        )
        cp = Checkpoint(
            dirname=f'{model_dir}',
            fn_prefix=f'{c}_'
        )
    
        # Load model
        net.initialize()
        net.load_params(checkpoint=cp)

        binary_clfs.append(net)

    # TODO this could be faster if swap to iterate over chunks instead of interating over classifiers (no redundant sample IO)
    dataset = datasets.ImageFolder(
        imagedir,
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]),
    )
    
    pred_prob = []
    for clf in tqdm(binary_clfs, desc='Detecting patterns'):
            pred_prob.append(clf.predict_proba(dataset)[:,1])
        
    # Save multiclass probability
    pred_prob = np.array(pred_prob)
    pred_prob = np.transpose(pred_prob)
    pred_prob = pd.DataFrame(pred_prob, columns=classes)
    pred_prob.index = pred_prob.index.astype(str)

    if 'sample_data' not in adata.uns:
        adata.uns['sample_data'] = dict()
        
    adata.uns['sample_data']['patterns_prob'] = pred_prob

    # Save mutliclass label
    pred_label = pred_prob.copy()
    pred_label[pred_prob >= 0.5] = 1
    pred_label[pred_prob < 0.5] = 0
    adata.uns['sample_data']['patterns'] = pred_label

    # Map pattern probabilities to obs
    # TODO fails with extracellular points
    # TODOD use _map_to_obs
    # for c in classes:
    #     adata.obs[f'{c}_prob'] = pred_prob[c][adata.obs['sample_id']].values
    #     adata.obs[f'{c}'] = pred_label[c][adata.obs['sample_id']].values


    return adata if copy else None

def get_conv_outsize(in_size, padding, dilation, kernel_size, stride):
    outsize = 1 + (in_size + 2*padding - dilation*(kernel_size-1) -1) / stride
    return int(outsize)

def temperature_scale(logits, temp):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    temperature = temp.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature

class SpotsModule(nn.Module):
    def __init__(self, params, eval_prob=False, temperature=1) -> None:
        super().__init__()
        self.eval_prob = eval_prob
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
#         n_conv_layers = params["n_convs"]
        n_conv_layers = 2
        conv_layers = []

        in_channels = 1
        in_dim = 32

        # Stack convolutions + batchnorm + activation
        for i in range(n_conv_layers):
            out_channels = 16
            kernel_size = 5
    
            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size
                )
            )
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2, 2))

            # Compute convolved output dimensions
            in_dim = get_conv_outsize(in_dim, padding=0, dilation=1, kernel_size=kernel_size, stride=1)
            
            in_dim = int(in_dim / 2)
            in_channels = out_channels

        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        fc_layers = [nn.Flatten()]

        # Compute flatten size
        in_features = out_channels * in_dim * in_dim
        n_fc_layers=2
        
        for i in range(n_fc_layers):
            out_features = params[f"n_units_l{i}"]
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.Dropout())
            fc_layers.append(nn.ReLU())

            in_features = out_features            
            
        fc_layers.append(nn.Linear(in_features, 2))

        self.model = nn.Sequential(*[*conv_layers, *fc_layers])

    def forward(self, x):
        x = self.model(x)
        
        x = temperature_scale(x, self.temperature)
        if self.eval_prob:
            x = F.softmax(x, dim=-1)
            
        return x