import pickle
import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append("../../")

import os
import numpy as np
import torch
from src.models.spektrankers import SVDRankerNormal, SVDRankerCov, SVDRankerKCov, SerialRank, CSerialRank, KCCARank, \
    CCARank
from models.spektrankle_misc import C_to_choix_ls
from models.baselines import BradleyTerryRanker, Pairwise_LogisticRegression
from models.prefkrr import PreferentialKRR

from data.load_experiments import unseen_setup, Pokemon
from models.spektrankle_misc import compute_upsets, median_heuristic
from gpytorch.kernels import RBFKernel, ScaleKernel


def extract_upsets(r, C):
    a, b, _ = compute_upsets(r, C, verbose=False)
    return max([a, b])


seed_ls = [i for i in range(20)]

if __name__ == '__main__':
    for seed in seed_ls:
        np.random.seed(seed)

        C_train, C_test, _, _, X, X_test, _ = unseen_setup(Pokemon(split=1), sparsity=0.7)

        d = X.shape[1]
        k = ScaleKernel(RBFKernel(ard_num_dims=d))
        k.lengthscalse = median_heuristic(X)
        K = k(torch.tensor(X).float()).evaluate().detach().numpy()
        K_test = k(torch.tensor(X), torch.tensor(X_test)).evaluate().detach().numpy().T

        # Result holder
        upset_train = dict()
        upset_test = dict()

        # SVDC
        svdc = SVDRankerCov(C_train, X, verbose=False)
        svdc.fit()
        svdc_pred = svdc.predict(X_test)

        upset_train["svdc"] = extract_upsets(svdc.r, C_train)
        upset_test["svdc"] = extract_upsets(svdc_pred, C_test)

        # SVDN
        svdn = SVDRankerNormal(C_train, verbose=False)
        svdn.fit()

        upset_train["svdn"] = extract_upsets(svdn.r, C_train)
        upset_test["svdn"] = 0

        # SVDK
        svdk = SVDRankerKCov(C_train, K, verbose=False)
        svdk.fit()
        svdk_pred = svdk.predict(K_test)

        upset_train["svdk"] = extract_upsets(svdk.r, C_train)
        upset_test["svdk"] = extract_upsets(svdk_pred, C_test)

        # Serial
        serial = SerialRank(C_train, verbose=False)
        serial.fit()

        upset_train["serial"] = extract_upsets(serial.r, C_train)
        upset_test["serial"] = 0

        # C-Serial
        cserial = CSerialRank(C_train, K, 1e-1, verbose=False)
        cserial.fit()
        cserial2 = CSerialRank(C_train, K, 1, verbose=False)
        cserial2.fit()

        upset_train["c-serial"] = max(extract_upsets(cserial.r, C_train), extract_upsets(cserial2.r, C_train))
        upset_test["c-serial"] = 0

        # CCA Rank
        cca = CCARank(C_train, X, verbose=False)
        cca.fit()

        upset_train["CCA"] = extract_upsets(cca.r, C_train)
        upset_test["CCA"] = extract_upsets(cca.predict(X_test), C_test)

        # KCCA RANK
        kcca = KCCARank(C_train, X, 1e-1, verbose=False)
        kcca.fit()

        upset_train["KCCA"] = extract_upsets(kcca.r, C_train)
        upset_test["KCCA"] = extract_upsets(kcca.predict(X_test), C_test)

        # BT
        bt = BradleyTerryRanker(C_train, verbose=False)
        bt.fit()

        upset_train["BT"] = extract_upsets(bt.r, C_train)
        upset_test["BT"] = 0

        # BT Log Reg
        chx_ls = C_to_choix_ls(C_train)
        y_ls = [1 for i in range(chx_ls.shape[0])]

        plr = Pairwise_LogisticRegression(chx_ls, y_ls, X)
        plr.fit()
        _, lr_test = plr.learning_to_rank(), plr.predict_unseen_rank(X_test)

        upset_train["PairLR"] = extract_upsets(plr.r, C_train)
        upset_test["PairLR"] = extract_upsets(lr_test, C_test)

        # Pref KRR
        prefkrr = PreferentialKRR(C_train, X)
        prefkrr.fit(epoch=500, lr=1e-1)
        f = prefkrr.f

        upset_train["pkrr"] = extract_upsets(f, C_train)
        upset_test["pkrr"] = extract_upsets(prefkrr.predict(K_test), C_test)

        results = [upset_train, upset_test]

        print(results)
        print("\n")

        job_name = 'pokemon_results'
        if not os.path.exists(job_name):
            os.makedirs(job_name)

        name = job_name + "/" + str(seed) + ".pkl"

        with open(name, "wb") as f:
            pickle.dump(results, f)

    print("DONE!")
