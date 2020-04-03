import numpy as np


def cart2pol(x, y):
    x = x.astype(np.float)
    y = y.astype(np.float)

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def read_as_10x(open_dir, cells):
    # TODO refactor
    print("Reading sparse matrix data")
    adatas = []
    for cell in tqdm(cells):
        adatas.append(sc.read_10x_mtx(open_dir + cell+'/data'))
    self.alldata = adatas[0].concatenate(
        adatas[1:], batch_categories=cells)
    self.alldata = desc.scale_bygroup(alldata, groupby="batch")


def run_desc(model_dir, GPU=False, num_Cores=1):
    # TODO refactor
    self.alldata = desc.train(alldata,
                              dims=[alldata.shape[1], 64, 32],
                              tol=0.005,
                              n_neighbors=50,
                              batch_size=1000,
                              # not necessarily a list, you can only set one value, like, louvain_resolution=1.0
                              louvain_resolution=[0.8, 1.0],
                              save_dir=str(model_dir),
                              do_tsne=True,
                              learning_rate=200,  # the parameter of tsne
                              use_GPU=GPU,
                              num_Cores=numCores,  # for reproducible, only use 1 cpu
                              num_Cores_tsne=4,
                              save_encoder_weights=False,
                              save_encoder_step=3,  # save_encoder_weights is False, this parameter is not used
                              use_ae_weights=False,
                              do_umap=False)  # if do_uamp is False, it will don't compute umap coordiate


def plot_desc():
    # TODO refactor
    sc.pl.scatter(alldata, basis='tsne', color=["desc_1.0", "batch"])
