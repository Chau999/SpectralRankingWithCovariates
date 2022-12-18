"""
Spectral Ranking experiments: Rank Prediction

Idea:

Using the ranking simulation I have, we use the corrupt comparison matrix to learn the ranking function,
and then compare it against test items with ranking that we know beforehand.

"""
import pickle
import warnings

warnings.filterwarnings("ignore")
import itertools
import os
import numpy as np
import torch
from src.models.spektrankers import SVDRankerNormal, SVDRankerNCov, SVDRankerKCov, SerialRank, CSerialRank, KCCARank, CCARank, \
    DiffusionRankCentrality, RankCentrality
from models.spektrankle_misc import median_heuristic, C_to_choix_ls, train_test_split_C, kendalltau_score
from gpytorch.kernels import RBFKernel
from data.simulation import Simulation
from models.baselines import BradleyTerryRanker, Pairwise_LogisticRegression
from models.prefkrr import PreferentialKRR


def main(n, d, sparsity_rate, error_rate, error_type, noise_level, train_ratio, seed, job_name):

    np.random.seed(seed)

    # Simulate matches
    sim = Simulation(n, d, sparsity_rate, p=error_rate, type=error_type)
    sim.generate(noise=noise_level)

    X = sim.X
    C = sim.C  # the full corrupted matrix

    C_train, _ = train_test_split_C(C, train_ratio=train_ratio)
    cut = round(n * train_ratio)

    test_r = sim.true_r[cut:]
    train_r = sim.true_r[:cut]

    train_kendall = dict()
    test_kendall = dict()

    # Fit models

    # SVDN
    svdn = SVDRankerNormal(C_train, verbose=False)
    svdn.fit()
    train_kendall["svdn"], test_kendall["svdn"] = kendalltau_score(train_r, svdn.r[:cut]), 0

    # SVDC
    svdc = SVDRankerNCov(C_train, X, verbose=False)
    svdc.fit()
    train_kendall["svdc"], test_kendall["svdc"] = kendalltau_score(train_r, svdc.r[:cut]), kendalltau_score(test_r,
                                                                                                            svdc.r[
                                                                                                            cut:])

    # SVDK
    k = RBFKernel(ard_num_dims=d)
    k.lengthscalse = median_heuristic(X)
    K = k(torch.tensor(X).float()).evaluate().detach().numpy()
    svdk = SVDRankerKCov(C_train, K, verbose=False)
    svdk.fit()
    train_kendall["svdk"], test_kendall["svdk"] = kendalltau_score(train_r, svdk.r[:cut]), kendalltau_score(test_r,
                                                                                                            svdk.r[
                                                                                                            cut:])

    # Serial Rank
    serial = SerialRank(C_train, verbose=False)
    serial.fit()
    train_kendall["serial"], test_kendall["serial"] = kendalltau_score(train_r, serial.r[:cut]), 0

    # C-Serial Rank
    cserial = CSerialRank(C_train, K, 1e-1, verbose=False)
    cserial.fit()

    train_score = kendalltau_score(train_r, cserial.r[:cut])
    test_score = kendalltau_score(test_r, cserial.r[cut:])

    train_kendall["c-serial"], test_kendall["c-serial"] = train_score, test_score

    # CCA Rank
    cca = CCARank(C_train, X)
    cca.fit()
    train_kendall["cca"], test_kendall["cca"] = kendalltau_score(train_r, cca.predict(X[:cut, :])), kendalltau_score(
        test_r, cca.predict(X[cut:, :]))

    # KCCA RANK
    kcca = KCCARank(C_train, X, 1e-1, lengthscale=median_heuristic(X, type="not torch"))
    kcca.fit()
    kcca_train_r, kcca_test_r = kcca.predict(X[:cut, :]), kcca.predict(X[cut:, :])
    train_kendall["kcca"], test_kendall["kcca"] = kendalltau_score(train_r, kcca_train_r), kendalltau_score(test_r,
                                                                                                            kcca_test_r)

    # Bradley Terry Ranker
    bt = BradleyTerryRanker(C_train, verbose=False)
    bt.fit()
    train_kendall["BT"], test_kendall["BT"] = kendalltau_score(train_r, bt.r[:cut]), 0

    # Pairwise LR
    chx_ls = C_to_choix_ls(C_train)
    y_ls = [1 for i in range(chx_ls.shape[0])]

    plr = Pairwise_LogisticRegression(chx_ls, y_ls, X)
    plr.fit()
    plr.learning_to_rank()
    train_kendall["PairLR"], test_kendall["PairLR"] = kendalltau_score(train_r, plr.r[:cut]), kendalltau_score(test_r,
                                                                                                               plr.r[
                                                                                                               cut:])

    # Pref KRR
    prefkrr = PreferentialKRR(C_train, X)
    prefkrr.fit(epoch=1000, lr=1e-2)
    f = prefkrr.f
    train_kendall["pkrr"], test_kendall["pkrr"] = kendalltau_score(train_r, f[:cut]), kendalltau_score(test_r, f[cut:])

    # DiffusionCentrality
    dc = DiffusionRankCentrality(C_train, K)
    dc.fit()
    train_kendall["dc"], test_kendall["dc"] = kendalltau_score(train_r, dc.r[:cut]), 0

    rc = RankCentrality(C_train)
    rc.fit()
    train_kendall["rc"], test_kendall["rc"] = kendalltau_score(train_r, rc.r[:cut]), 0

    # Print Results
    result_dict = {"n": n,
                   "d": d,
                   "sparsity_rate": sparsity_rate,
                   "error_rate": error_rate,
                   "error_type": error_type,
                   "noise_level": noise_level,
                   "train_ratio": train_ratio,
                   "seed": seed,
                   "train_kendall": train_kendall,
                   "test_kendall": test_kendall}

    name = "simulation_n{}_d{}_sparsity{}_error_rate{}_error_type{}_noise{}_split{}_seed{}.pkl".format(n,
                                                                                                       d,
                                                                                                       sparsity_rate,
                                                                                                       error_rate,
                                                                                                       error_type,
                                                                                                       noise_level,
                                                                                                       train_ratio,
                                                                                                       seed
                                                                                                       )
    with open(job_name + "/" + name, "wb") as f:
        pickle.dump(result_dict, f)


def slurm_parallel_main(args, job_name):
    n, d, sparsity_rate, error_rate, error_type, noise_level, train_ratio, seed = args
    main(n, d, sparsity_rate, error_rate, error_type, noise_level, train_ratio, seed, job_name)


if __name__ == '__main__':

    # Create your path
    job_name = 'simulations'
    if not os.path.exists(job_name):
        os.makedirs(job_name)

    n_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    print('Using' + str(n_cpus) + ' CPUS per job array task')
    slurm_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
    slurm_parameter = int(os.environ['SLURM_ARRAY_TASK_ID'])

    n_ls = [1000]
    d_ls = [1]
    sparsity_rate_ls = [0.7, 0.9]
    error_rate_ls = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    error_type_ls = ["FLIP"]
    noise_level_ls = [0, 0.05, 0.1, 0.2, 0.5]
    train_ratio_ls = [0.7]
    seed_ls = [int(i) for i in range(20)]


    vector_grid = list(itertools.product(n_ls,
                                         d_ls,
                                         sparsity_rate_ls,
                                         error_rate_ls,
                                         error_type_ls,
                                         noise_level_ls,
                                         train_ratio_ls,
                                         seed_ls)
                       )

    parameter = vector_grid[int(slurm_parameter)]
    slurm_parallel_main(args=parameter, job_name=job_name)