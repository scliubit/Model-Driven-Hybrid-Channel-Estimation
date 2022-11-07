from LAMP_CE_Network import load_trainable_vars
import numpy as np
import tensorflow as tf
import myshrinkage
from scipy.io import savemat, loadmat
from os import path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # BE QUIET!!!!  设置log日志级别，只显示warning和error
import math
pi = math.pi

shrink = 'bg'  # 'soft'
type = 'ULA'  # 'UPA'
MN = '40256'
M = 40
N = 256
G = 1024
K = 64
L = 8

SNRrange = [0, 5, 10, 15, 20]
# SNRrange = [15, 20]


def getNoise(y, snr):
    ypower = tf.reduce_mean(tf.square(tf.abs(y)), axis=0, keepdims=True)
    noise_var = tf.cast(ypower / ((10 ** (snr / 10)) * 2), tf.complex128)
    noise = tf.complex(real=tf.random_normal(shape=tf.shape(y), dtype=tf.float64),
                       imag=tf.random_normal(shape=tf.shape(y), dtype=tf.float64))
    n = tf.sqrt(noise_var) * noise

    return n


D = loadmat(type + 'testdata' + str(N) + '_' + str(K) + '.mat')
ht = D['h']
ht = np.transpose(ht, [2, 0, 1])
U1 = D['U']
for snr in range(len(SNRrange)):
    # get the trained network
    trainedfilename = type + '_' + shrink + '_' + MN + '_train' + str(64) + 'carriers' + str(30) + 'dB.mat'
    # trainedfilename = type + '_' + shrink + '_' + MN + str(SNRrange[snr]) + 'dB.mat'
    saveresultsname = type + 'results' + '_' + shrink + '_' + MN + '_train64carriers_test' + str(K) + 'carriers.mat'
    # saveresultsname = type + 'results' + '_' + shrink + '_' + MN + '_' + str(N_bits) + 'bits_quan_phase' + '.mat'

    if not path.exists(saveresultsname):
        # print('There is no file for saving the result before, and it has been generated!')
        if shrink is 'bg':
            LAMP_nmse = np.zeros(dtype=np.float64, shape=(len(SNRrange), 1))
            D = dict(LAMP_nmse=LAMP_nmse)
            savemat(saveresultsname, D)
        else:
            if shrink is 'gm':
                GMLAMP_nmse = np.zeros(dtype=np.float64, shape=(5, 1))
                D = dict(GMLAMP_nmse=GMLAMP_nmse)
                savemat(saveresultsname, D)

    T = 10
    untied = False

    eta, theta_init = myshrinkage.get_shrinkage_function(shrink)
    var_all = []
    OneOverM = tf.constant(float(1) / M, dtype=tf.float64)
    Atheta = tf.random_uniform(shape=(M, N), minval=0, maxval=2*pi, dtype=tf.float64)
    Atheta_ = tf.Variable(initial_value=Atheta, name='Atheta_' + str(30) + '_0')
    var_all.append(Atheta_)
    # q = tf.round(Atheta_ * N_Bits / (2 * pi))
    # Atheta_quan = q * 2 * pi / N_Bits
    Areal = tf.multiply(tf.cos(Atheta_), tf.sqrt(OneOverM))
    Aimag = tf.multiply(tf.sin(Atheta_), tf.sqrt(OneOverM))
    A_ = tf.complex(Areal, Aimag, name='A')  # 恒模预编码矩阵

    h_ = tf.placeholder(tf.complex128, (None, N, K))
    h1 = tf.transpose(h_, [1, 0, 2])
    y1_ = []
    for k_1 in range(K):
        y1 = tf.matmul(A_, h1[:, :, k_1])
        noise = getNoise(y1, SNRrange[snr])
        y1 = y1 + noise
        y1_.append(y1)
    ytemp_ = tf.stack(y1_, axis=2)
    y_ = tf.transpose(ytemp_,  [1, 0, 2])

    v_ = y_
    OneOverMK = tf.constant(float(1) / (M*K), dtype=tf.float64)
    rvar_ = OneOverMK * tf.expand_dims(tf.square(tf.norm(tf.abs(v_), axis=[1, 2])), 1)
    U_ = tf.placeholder(tf.complex128, (G, N))
    A_H = tf.transpose(A_, conjugate=True)
    B = tf.matmul(U_, A_H)  # 初始化B
    Breal_ = tf.Variable(tf.real(B), name='Breal_' + str(30) + '_1')
    var_all.append(Breal_)
    Bimag_ = tf.Variable(tf.imag(B), name='Bimag_' + str(30) + '_1')
    var_all.append(Bimag_)
    B_ = tf.complex(Breal_, Bimag_, name='B')
    v1 = tf.transpose(v_, [1, 0, 2])
    v1 = tf.reshape(v1, (M, -1))
    Bvtemp = tf.matmul(B_, v1)
    Bv = tf.reshape(Bvtemp, (G, tf.shape(h_)[0], K))
    Bv_ = tf.transpose(Bv, [1, 0, 2])
    theta_ = tf.Variable(theta_init, dtype=tf.float64, name='theta_' + str(30) + '_1')
    var_all.append(theta_)
    xhat_, dxdr_ = eta(Bv_, rvar_, K, theta_)
    GOverM = tf.constant(float(G) / M, dtype=tf.complex128)
    xhat_, dxdr_ = eta(Bv_, rvar_, K, theta_)
    for t in range(2, T+1):
        b_ = tf.expand_dims(GOverM * dxdr_, 1)
        U_H = tf.transpose(U_, conjugate=True)
        matrix = tf.matmul(A_, U_H)
        x2 = tf.transpose(xhat_, [1, 0, 2])
        x3 = tf.reshape(x2, (G, -1))
        Axhat = tf.matmul(matrix, x3)
        Axhat = tf.reshape(Axhat, (M, tf.shape(h_)[0], K))
        Axhat_ = tf.transpose(Axhat, [1, 0, 2])
        v_ = tf.reshape(v_, [tf.shape(h_)[0], M*K])
        bv = tf.multiply(b_, v_)
        bv_ = tf.reshape(bv, [tf.shape(h_)[0], M, K])
        v_ = y_ - Axhat_ + bv_
        rvar_ = OneOverMK * tf.expand_dims(tf.square(tf.norm(tf.abs(v_), axis=[1, 2])), 1)

        if untied:  # 表明每一层的B都会训练
            Breal_ = tf.Variable(tf.real(B), name='Breal_' + str(snr) + '_' + str(t))
            var_all.append(Breal_)
            Bimag_ = tf.Variable(tf.imag(B), name='Bimag_' + str(snr) + '_' + str(t))
            var_all.append(Bimag_)
            B_ = tf.complex(Breal_, Bimag_, name='B')
            Bv_ = tf.matmul(B_, v_)
            rhat_ = xhat_ + Bv_
        else:
            v3 = tf.transpose(v_, [1, 0, 2])
            v4 = tf.reshape(v3, (M, -1))
            Bv = tf.matmul(B_, v4)
            Bv = tf.reshape(Bv, (G, tf.shape(h_)[0], K))
            Bv_ = tf.transpose(Bv, [1, 0, 2])
            rhat_ = xhat_ + Bv_
        xhat_, dxdr_ = eta(rhat_, rvar_, K, theta_)
    U_H = tf.transpose(U_, conjugate=True)
    xhat1 = tf.transpose(xhat_, [1, 0, 2])
    xhat2 = tf.reshape(xhat1, (G, -1))
    hhat = tf.matmul(U_H, xhat2)
    hhat = tf.reshape(hhat, (N, tf.shape(h_)[0], K))
    hhat_ = tf.transpose(hhat, [1, 0, 2])
    nmse_ = tf.reduce_mean(
        tf.square(tf.norm(tf.abs(hhat_ - h_), axis=[1, 2])) / tf.square(tf.norm(tf.abs(h_), axis=[1, 2])))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer(), feed_dict={U_: U1})
    load_trainable_vars(sess, trainedfilename)

    nmse_SNR = []

    xhat = sess.run(xhat_, feed_dict={h_: ht, U_: U1})
    nmse = sess.run(nmse_, feed_dict={h_: ht, U_: U1})
    nmse_dB = 10 * np.log10(nmse)
    print(str(SNRrange[snr]) + 'dB:' + ' ' + 'NMSE = ' + str(nmse_dB) + 'dB')
    nmse_SNR = np.append(nmse_SNR, nmse_dB)
    # print(nmse_SNR)
    results = loadmat(saveresultsname)
    if shrink is 'gm':
        GMLAMP_nmse = results['GMLAMP_nmse']
        # GMLAMP_nmse = GMLAMP_nmse[0]
        GMLAMP_nmse = np.append(GMLAMP_nmse, nmse_SNR)
        # GMLAMP_nmse[ibegin:iend] = nmse_SNR
        print(GMLAMP_nmse)
        D = dict(GMLAMP_nmse=GMLAMP_nmse)
        savemat(saveresultsname, D)
    else:
        if shrink is 'bg':
            # print(results)
            LAMP_nmse = results['LAMP_nmse']
            LAMP_nmse[snr, :] = nmse_SNR
            D = dict(LAMP_nmse=LAMP_nmse)
            savemat(saveresultsname, D)
    tf.reset_default_graph()
