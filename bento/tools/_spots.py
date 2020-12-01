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

from .._settings import settings

def spots(data, imagedir, device='auto', copy=False):
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

    dataset = datasets.ImageFolder(
        imagedir,
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]),
    )
    
    # Default to gpu if possible. Otherwise respect specified parameter
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_dir = '/'.join(bento.__file__.split('/')[:-1]) + '/models/spots'

    # Load temperature values as list
    with open(f'{model_dir}/temperature.txt', 'r') as f:
        temperatures = f.read().splitlines()

    # Load ensemble model
    binary_clfs = []
    for i, c in enumerate(classes):

        net = NeuralNetClassifier(
            module=SpotsNet,
            module__eval_prob=True,
            module__temperature=temperatures[i]
        )

        net.initialize()
        cp = Checkpoint(dirname=f'{model_dir}', fn_prefix=f'{c}_')
        net.load_params(checkpoint=cp)
        binary_clfs.append(net)

    # TODO this could be faster if swap to iterate over chunks instead of interating over classifiers (no redundant sample IO)
    pred_prob = []
    for clf in tqdm(binary_clfs, desc='Detecting patterns'):

        if settings.n_cores > 1:
            parallel = Parallel(n_jobs=settings.n_cores, max_nbytes=None)
            results = parallel(delayed(clf.predict_proba)(chunk) for chunk in np.array_split(X, settings.n_cores))
            pred_prob.append(np.vstack(results)[:,1].flatten())
        else:
            pred_prob.append(clf.predict_proba(dataset)[:,1])
        
    # Save multiclass probability
    pred_prob = np.array(pred_prob)
    pred_prob = np.transpose(pred_prob)
    pred_prob = pd.DataFrame(pred_prob, columns=classes)
    pred_prob.index = pred_prob.astype(str)
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

class SpotsNet(nn.Module):
    '''
    3 2d-convolutional layers with dilation to increase receptive field size
    ''' 
    def __init__(self, eval_prob=False, temperature=1):
        super(SpotsNet, self).__init__()
        
        self.eval_prob = eval_prob
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.kernel_size1 = 5
        self.dilation_size1 = 1

        self.kernel_size2 = 5
        self.dilation_size2 = 2

        # Convolution block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=self.kernel_size1, dilation=self.dilation_size1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Convolution block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=self.kernel_size2, dilation=self.dilation_size2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Calculate total size to flatten to FC layer
        conv1_outsize = get_conv_outsize(64, 0, self.dilation_size1, self.kernel_size1, 1)
        conv2_outsize = get_conv_outsize(conv1_outsize / 2, 0, self.dilation_size2, self.kernel_size2, 1)
        fc1_input_size = int(16 * (conv2_outsize / 2) * (conv2_outsize / 2))
        
        # First FC layer flattens convolutions
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc1_input_size, 128),
            nn.Dropout(),
            nn.ReLU()
        )
        
        # Second FC layer
        self.fc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.Dropout(),
            nn.ReLU()
        )
        
        # 3rd FC layer half size -> binary class probability with softmax
        self.fc3 = nn.Sequential(
            nn.Linear(32, 2)
        )
        
    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = temperature_scale(x, self.temperature)
        if self.eval_prob:
            x = F.softmax(x, dim=-1)
        return x