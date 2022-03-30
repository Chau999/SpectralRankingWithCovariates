import copy
import numpy as np

import torch
from numpy.random import rand, randn, randint
from numpy import sign, sqrt, count_nonzero, cos, sin, pi, ones, zeros, mean, shape, reshape, exp, sort, diag, eye, around, \
    linspace, dot, sqrt, argsort, allclose, bool, fill_diagonal, concatenate, empty, arange
from matplotlib.pyplot import plot, show, imshow
from numpy.linalg import svd, solve, inv, eig, eigh, cholesky, norm

from sklearn.metrics import pairwise_distances


def sqrtm_psd(A, check_finite=True):
    """
    Returns the matrix square root of a positive semidefinite matrix,
    truncating negative eigenvalues.
    """
    s, V = eigh(A)
    s[s <= 0] = 0
    s = sqrt(s)
    A_sqrt = (V * s).dot(V.conj().T)
    return A_sqrt


def invsqrtm_psd(A, check_finite=True):
    """
    Returns the inverse matrix square root of a positive semidefinite matrix,
    truncating negative eigenvalues.
    """
    s, V = eigh(A)
    s[s <= 1e-16] = 0
    s[s > 0] = 1/sqrt(s[s > 0])
    A_invsqrt = (V * s).dot(V.conj().T)
    return A_invsqrt


def centering_matrix(n):
    # centering matrix, projection to the subspace orthogonal
    # to all-ones vector
    return eye(n)-ones((n, n))/n


def rank_items(array):
    temp = array.argsort()
    ranks = empty(len(array), int)
    ranks[temp] = arange(len(array))
    return ranks+1


def get_the_subspace_basis(n, verbose=True):
    # returns the orthonormal basis of the subspace orthogonal
    # to all-ones vector
    H = centering_matrix(n)
    s, Zp = eigh(H)
    ind = argsort(-s)  # order eigenvalues descending
    s = s[ind]
    Zp = Zp[:, ind]   # second axis !!
    if(verbose):
        print("...forming the Z-basis")
        print("check eigenvalues: ", allclose(
            s, concatenate((ones(n-1), [0]), 0)))

    Z = Zp[:, :(n-1)]
    if(verbose):
        print("check ZZ'=H: ", allclose(dot(Z, Z.T), H))
        print("check Z'Z=I: ", allclose(dot(Z.T, Z), eye(n-1)))
    return Z


def compute_upsets(r, C, verbose=True, which_method=""):
    n = shape(r)[0]
    totmatches = count_nonzero(C)/2
    if(len(shape(r)) == 1):
        r = reshape(r, (n, 1))
    e = ones((n, 1))
    Chat = r.dot(e.T)-e.dot(r.T)
    upsetsplus = count_nonzero(sign(Chat[C != 0]) != sign(C[C != 0]))
    upsetsminus = count_nonzero(sign(-Chat[C != 0]) != sign(C[C != 0]))
    winsign = 2*(upsetsplus < upsetsminus)-1
    if(verbose):
        print(which_method+" upsets(+): %.4f" %
              (upsetsplus/float(2*totmatches)))
        print(which_method+" upsets(-): %.4f" %
              (upsetsminus/float(2*totmatches)))
    return upsetsplus/float(2*totmatches), upsetsminus/float(2*totmatches), winsign


def Compute_Sim(C):
    """
    Compute the Similarity matrix
    """
    n = C.shape[0]
    ones_mat = n * np.dot(np.ones(n).reshape(-1, 1), np.ones(n).reshape(1, -1))
    S = 0.5 * (ones_mat + np.dot(C, C.T))
    return S


def GraphLaplacian(G):
    """
    Input a simlarity graph G and return graph GraphLaplacian
    """
    D = np.diag(G.sum(axis=1))
    L = D - G

    return L


def Signed_GraphLaplacian(G):
    """
    Compute the signed graph laplacian of a signed graph
    """

    D = np.diag(np.abs(G).sum(axis=1))
    L = D - G

    return L


def Separate_Constraints(Q):
    """
    Separate the constraints of the given information
    Return: a matrix of positive constraints that you want to encourage
            another matrix of negative constraints that you want to discourage
    """
    Q_plus = copy.deepcopy(Q)
    Q_minus = copy.deepcopy(Q)
    Q_plus[Q < 0] = 0
    Q_minus[Q > 0] = 0

    return Q_plus, -1 * Q_minus


def median_heuristic(X, type="torch"):

    if X.shape[1] == 1:
        lengthscales = np.median(pairwise_distances(X))
    else:
        lengthscales = []
        for j in range(X.shape[1]):
            lengthscales.append(np.median(pairwise_distances(X[:, [j]])))

    if type == "torch":
        return torch.tensor(lengthscales)
    else:
        return lengthscales


def C_to_choix_ls(C):

    win_matches = np.nonzero(C > 0)
    return np.concatenate((win_matches[0].reshape(-1, 1), win_matches[1].reshape(-1, 1)), axis=1)


def train_test_split_C(C, train_ratio=0.7):

    C_train, C_test = np.zeros_like(C), np.zeros_like(C)
    cut = round(C.shape[0]*train_ratio)
    C_train[:cut,:cut] = C[:cut, :cut]
    C_test[cut:, cut:] = C[cut:, cut:]

    return C_train, C_test