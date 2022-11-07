__author__ = 'WEI'

# import tensorflow as tf
import numpy as np
# import numpy.linalg as la
# import multiprocessing as mp
from scipy.io import savemat
# from scipy.io import loadmat
import math

pi = math.pi


class Problem(object):
    def __init__(self, N_BS, G, K, L, fs, fc, tau_max, maxiter):
        self.N_BS = N_BS
        self.G = G
        self.K = K
        self.L = L
        self.fs = fs
        self.fc = fc
        self.tau_max = tau_max
        self.maxiter = maxiter

    def genU(self):
        N_BS = self.N_BS
        G = self.G
        U = np.zeros(dtype=np.complex128, shape=(G, N_BS))
        Nidex = np.arange(0, N_BS)
        A = np.array(Nidex)
        A = A.reshape((1, len(A)))

        index = 0
        for i in range(-(G - 1), G, 2):
            i = i / (2 * G)
            U[index, ...] = (1 / np.sqrt(N_BS)) * np.exp(-1j * 2 * pi * A * i)
            index = index + 1
        return U

    def genChannel(self):
        N_BS = self.N_BS
        G= self.G
        K = self.K
        L = self.L
        fs = self.fs
        fc = self.fc
        tau_max = self.tau_max
        maxiter = self.maxiter
        H = np.zeros(dtype=np.complex128, shape=(N_BS, K))
        subH = np.zeros(dtype=np.complex128, shape=(N_BS, L))
        U = self.genU()

        Nidex = np.arange(0, N_BS)
        A = np.array(Nidex)
        A = A.reshape((1, len(A)))

        for l in range(L):
            alpha = (np.random.normal() + 1j * np.random.normal()) / np.sqrt(2)  # 增益
            theta = pi * np.random.uniform(0, 1) - pi / 2  # 角度
            a_theta = (1 / np.sqrt(N_BS)) * np.exp(1j * 2 * pi * A * np.sin(theta) / 2)  # 阵列响应(天线,路径）
            subH[..., l] = alpha * a_theta
        # delay gain
        tau = tau_max * np.random.rand(1, L)
        tau = np.sort(tau)
        miu_tau = 2 * pi * tau * fs / K

        alpha_temp = np.sqrt(1 / 2) * (np.random.randn(1, L) + 1j * np.random.randn(1, L))
        alpha = np.sort(-alpha_temp)
        alpha = -alpha

        for k in range(K):
            D_diag = np.sqrt(N_BS / L) * np.diag(np.squeeze(alpha * np.exp(1j * k * miu_tau)))
            a = np.ones((L, 1))
            b = np.matmul(np.matmul(subH, D_diag), a)
            H[..., k] = np.squeeze(np.matmul(np.matmul(subH, D_diag), a))

        return H

    def genH(self):
        self.N_BS = N_BS
        self.G = G
        self.K = K
        self.L = L
        self.fs = fs
        self.fc = fc
        self.tau_max = tau_max
        self.maxiter = maxiter
        np.random.seed()
        H = np.zeros(dtype=np.complex128, shape=(N_BS, K, maxiter))
        H_nn = np.zeros(dtype=np.complex128, shape=(N_BS, maxiter*K))
        H_a = np.zeros(dtype=np.complex128, shape=(G, K, maxiter))
        H_a_nn = np.zeros(dtype=np.complex128, shape=(G, maxiter*K))
        U = self.genU()
        for i in range(maxiter):
            print(str(i) + '/' + str(maxiter))
            channel = self.genChannel()
            H[..., i] = channel
            H_nn[..., i*K:((i+1)*K)] = channel
            sparse_channel = np.matmul(U, channel)
            H_a[..., i] = sparse_channel
            H_a_nn[..., i*K:((i+1)*K)] = sparse_channel
            # savemat('example.mat', mdict={'test': H_nn})

        return H, H_nn, H_a, H_a_nn, U


class Generator(object):
    def __init__(self, **kwargs):
        vars(self).update(kwargs)


class NumpyGenerator(Generator):
    def __init__(self, **kwargs):
        Generator.__init__(self, **kwargs)


def genProblem(N_BS, G, K, L, fs, fc, tau_max, maxiter):
    opts = dict(N_BS=N_BS, G=G, K=K, L=L, fs=fs, fc=fc, tau_max=tau_max, maxiter=maxiter)
    p = Problem(**opts)

    prob = NumpyGenerator(p=p, opts=opts, iid=False)
    prob.H, prob.H_nn, prob.H_a, prob.H_a_nn, prob.U = p.genH()

    return prob


def save_problem(base, prob):
    print('saving {b}.mat'.format(b=base))
    # D = dict(h=prob.H, h_nn=prob.H_nn, h_a=prob.H_a, h_a_nn=prob.H_a_nn, U=prob.U)
    D = dict(h=prob.H, U=prob.U, h_nn=prob.H_nn)
    # np.savez( base + '.npz', **D)
    savemat(base + '.mat', D, oned_as='column')


if __name__ == '__main__':

    N_BS = 256  # 基站天线数
    G = 1024
    K = 64   # 载波数
    L = 8    # 路径数
    maxiter = 1000  # iter_number
    BW = 30.72e6  # 系统带宽
    fs = BW
    fc = 30e9
    tau_max = 0.2e-6
    # The total number of generated samples is K*nbatches
    genPro = genProblem(N_BS, G, K, L, fs, fc, tau_max, maxiter)
    save_problem('ULAtestdata' + str(N_BS) + '_' + str(K), genPro)

    # You can also generate the ULA training data and validation data by this file.
