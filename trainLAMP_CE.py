import LAMP_CE_Network
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # BE QUIET!!!!  设置log日志级别，只显示warning和error
type = 'ULA'  # 'UPA'
shrink = 'bg'  # 'soft'
MN = '40256'
M = 40
N = 256
G = 1024
K = 64
L = 8
T = 5
# SNRrange = [0, 5, 10, 15, 20]
SNRrange = [40]


for snr in SNRrange:
    # savenetworkfilename = type + '_' + shrink + '_' + str(snr) + 'dB.mat'
    savenetworkfilename = type + '_' + shrink + '_' + MN + '_train' + str(K) + 'carriers' + str(snr) + 'dB.mat'
    trainingfilename = type + 'traindata' + str(N) + '_' + str(K) + '.mat'
    validationfilename = type + 'validationdata' + str(N) + '_' + str(K) + '.mat'
    layers, h_, U = LAMP_CE_Network.build_LAMP(M=M, N=N, G=G, K=K, L=L, snr=snr, T=T, shrink=shrink, untied=False)
    training_stages = LAMP_CE_Network.setup_training(layers, h_, U, G, K, N, M, trinit=1e-3, refinements=(.5,))
    sess = LAMP_CE_Network.do_training(h_=h_, training_stages=training_stages, U=U, savefile=savenetworkfilename,
                                       trainingfile=trainingfilename, validationfile=validationfilename, maxit=1000000,
                                       snr=snr)
