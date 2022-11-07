import tensorflow as tf
import numpy as np
import myshrinkage
import sys
from scipy.io import loadmat, savemat
import math
pi = math.pi


def genNoise(y, snr):
    ypower = tf.reduce_mean(tf.square(tf.abs(y)), axis=0, keepdims=True)
    noise_var = tf.cast(ypower / ((10 ** (snr / 10)) * 2), tf.complex128)
    noise = tf.complex(real=tf.random_normal(shape=tf.shape(y), dtype=tf.float64),
                       imag=tf.random_normal(shape=tf.shape(y), dtype=tf.float64))
    n = tf.sqrt(noise_var) * noise

    return n


def build_LAMP(M, N, G, K, L, snr, T, shrink, untied):
    eta, theta_init = myshrinkage.get_shrinkage_function(shrink)
    layer = []
    var_all = []
    OneOverM = tf.constant(float(1) / M, dtype=tf.float64)
    Atheta = tf.random_uniform(shape=(M, N), minval=0, maxval=2*pi, dtype=tf.float64)
    Atheta_ = tf.Variable(initial_value=Atheta, name='Atheta_' + str(snr) + '_0')
    var_all.append(Atheta_)
    Areal = tf.multiply(tf.cos(Atheta_), tf.sqrt(OneOverM))
    Aimag = tf.multiply(tf.sin(Atheta_), tf.sqrt(OneOverM))
    A_ = tf.complex(Areal, Aimag, name='A')  # 恒模预编码矩阵

    h_ = tf.placeholder(tf.complex128, (None, N, K))  # 输入H
    # h1 = tf.transpose(h_, [1, 0, 2])
    # y1_ = []
    # for k_1 in range(K):
    #     y1 = tf.matmul(A_, h1[:, :, k_1])
    #     noise = genNoise(y1, snr)
    #     y1 = y1 + noise
    #     y1_.append(y1)
    # ytemp1 = tf.stack(y1_, axis=2)
    # ytemp_ = tf.transpose(ytemp1,  [1, 0, 2])
    h1 = tf.transpose(h_, [1, 0, 2])
    h1 = tf.reshape(h1, (N, -1))
    ytemp1 = tf.matmul(A_, h1)
    ytemp1 = tf.reshape(ytemp1, (M, tf.shape(h_)[0], K))
    ytemp_ = tf.transpose(ytemp1, [1, 0, 2])
    y_ = ytemp_
    # first layer 初始化v0=0，h0=0，故v1=y
    v_ = y_  # 残差为y

    OneOverMK = tf.constant(float(1) / (M*K), dtype=tf.float64)
    rvar_ = OneOverMK * tf.expand_dims(tf.square(tf.norm(tf.abs(v_), axis=[1, 2])), 1)

    U_ = tf.placeholder(tf.complex128, (G, N))
    A_H = tf.transpose(A_, conjugate=True)
    B = tf.matmul(U_, A_H)  # 初始化B
    Breal_ = tf.Variable(tf.real(B), name='Breal_' + str(snr) + '_1')
    var_all.append(Breal_)
    Bimag_ = tf.Variable(tf.imag(B), name='Bimag_' + str(snr) + '_1')
    var_all.append(Bimag_)
    B_ = tf.complex(Breal_, Bimag_, name='B')
    v1 = tf.transpose(v_, [1, 0, 2])
    v1 = tf.reshape(v1, (M, -1))
    Bvtemp = tf.matmul(B_, v1)
    Bv = tf.reshape(Bvtemp, (G, tf.shape(h_)[0], K))
    Bv_ = tf.transpose(Bv, [1, 0, 2])
    # x_gain = tf.reduce_sum(tf.square(tf.abs(Bv_)), axis=2)
    theta_ = tf.Variable(theta_init, dtype=tf.float64, name='theta_' + str(snr) + '_1')
    # theta_ = tf.expand_dims(theta_, 0)
    var_all.append(theta_)

    xhat_, dxdr_ = eta(Bv_, rvar_, K, theta_)
    # NOverM = tf.constant(float(N) / M, dtype=tf.complex128)
    GOverM = tf.constant(float(G) / M, dtype=tf.complex128)
    # layer.append(('LAMP-{0} linear T=1'.format(shrink), Bv_, (Breal_, Bimag_), tuple(var_all), (0,)))
    # layer.append(('LAMP-{0} non-linear T=1'.format(shrink), xhat_, (theta_,), tuple(var_all), (1,)))
    layer.append(('LAMP-{0} T=1'.format(shrink), xhat_, tuple(var_all), tuple(var_all), (0,)))

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
        # theta_ = tf.Variable(theta_init, dtype=tf.float64, name='theta_' + str(snr) + '_1')
        # var_all.append(theta_)
        xhat_, dxdr_ = eta(rhat_, rvar_, K, theta_)

        # layer.append(('LAMP-{0} linear T={1}'.format(shrink, t), rhat_, (Breal_, Bimag_), tuple(var_all), (0,)))
        # layer.append(('LAMP-{0} non-linear T={1}'.format(shrink, t), xhat_, (theta_,), tuple(var_all), (1,)))
        layer.append(('LAMP-{0} T={1}'.format(shrink, t), xhat_, tuple(var_all), tuple(var_all), (0,)))
    return layer, h_, U_


def setup_training(layers, x_, U, G, K, N, M, trinit=1e-3, refinements=(.5, .1, .01)):
    training_stages = []
    for name, xhat_, var_list, var_all, flag in layers:
        U_H = tf.transpose(U, conjugate=True)
        xhat1 = tf.transpose(xhat_, [1, 0, 2])
        xhat2 = tf.reshape(xhat1, (G, -1))
        hhat = tf.matmul(U_H, xhat2)
        hhat = tf.reshape(hhat, (N, tf.shape(x_)[0], K))
        hhat_ = tf.transpose(hhat, [1, 0, 2])
        # a = tf.square(tf.norm(tf.abs(hhat_ - x_), axis=[0, 1]))
        # b = tf.square(tf.norm(tf.abs(x_), axis=[0, 1]))
        nmse_ = tf.reduce_mean(
            tf.square(tf.norm(tf.abs(hhat_ - x_), axis=[1, 2])) / tf.square(tf.norm(tf.abs(x_), axis=[1, 2])))
        loss_ = nmse_
        print(var_list)
        if var_list is not None:
            if flag == (0,):
                train_ = tf.train.AdamOptimizer(trinit).minimize(loss_, var_list=var_list)
                training_stages.append((name, hhat_, loss_, nmse_, train_, var_list, var_all, flag))
            elif flag == (1,):
                train_ = tf.train.AdamOptimizer(trinit).minimize(loss_, var_list=var_list)
                training_stages.append((name, hhat_, loss_, nmse_, train_, var_list, var_all, flag))
            else:
                train_ = tf.train.AdamOptimizer(trinit).minimize(loss_, var_list=var_list)
                training_stages.append((name, hhat_, loss_, nmse_, train_, var_list, var_all, flag))
        index = 0
        for fm in refinements:
            train2_ = tf.train.AdamOptimizer(fm * trinit).minimize(loss_, var_list=var_all)
            training_stages.append((name + ' trainrate=' + str(index), hhat_, loss_, nmse_, train2_, (), var_all, flag))
            index = index + 1

    return training_stages


def load_trainable_vars(sess, filename):
    other = {}
    try:
        variables = tf.trainable_variables()
        tv = dict([(str(v.name).replace(':', '_'), v) for v in variables])
        for k, d in loadmat(filename).items():  # (k, d)表示字典中的(键，值)
            if k in tv:
                print('restore ' + k)
                sess.run(tf.assign(tv[k], d))
                # print(sess.run(tv[k]))
            else:
                if k == 'done':
                    for i in range(0, len(d)):
                        a = d[i]
                        d[i] = a.strip()
                other[k] = d
                # print('error!')
    except IOError:
        pass
    return other


def save_trainable_vars(sess, filename, snr, **kwargs):
    save = {}
    for v in tf.trainable_variables():
        if str(v.name).split('_')[1] == str(snr):
            save[str(v.name).replace(':', '_')] = sess.run(v)
        continue
        # save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    savemat(filename, save)
    # np.savez(filename, **save)


def assign_trainable_vars(sess, var_list, var_list_old):
    for i in range(len(var_list)):
        temp = sess.run(var_list_old[i])
        # print(temp)
        sess.run(tf.assign(var_list[i], temp))


def do_training(h_, training_stages, U, savefile, trainingfile, validationfile, snr, iv1=10, maxit=1000000,
                better_wait=5000):
    Dtraining = loadmat(trainingfile)
    ht = Dtraining['h']
    # hat = Dtraining['h_a']
    U1 = Dtraining['U']
    trainingsize = np.size(ht, axis=2)
    Dvalidation = loadmat(validationfile)
    hv = Dvalidation['h']
    hv = np.transpose(hv, [2, 0, 1])
    # hav = Dvalidation['h']

    sess = tf.Session()
    sess.run(tf.global_variables_initializer(), feed_dict={U: U1})

    state = load_trainable_vars(sess, savefile)
    # state = []
    done = state.get('done', [])
    log = state.get('log', [])
    layernmse = state.get('layernmse', [])

    var_list_old0 = ()  # B
    var_list_old1 = ()  # theta
    var_list_old2 = ()  # Atheta
    nmse_dB = None
    for name, xhat_, loss_, nmse_, train_, var_list, var_all, flag in training_stages:
        if name in done:
            if name == 'LAMP-gm linear T=5':
                var_list_old0 = var_list
            if name == 'LAMP-gm non-linear T=4':
                var_list_old1 = var_list
            print('Already did  ' + name + ' skipping.')
            continue
        if len(var_list):
            print('')
            print(name + ' ' + 'extending ' + ','.join([v.name for v in var_list]))
            if flag == (0,):  # if linear operation
                if nmse_dB is not None:
                    layernmse = np.append(layernmse, nmse_dB)
                    print(layernmse)
                if len(var_list_old0):
                    # Initialize the training variable to the value of that in previous layer
                    assign_trainable_vars(sess, var_list, var_list_old0)
                    # print(var_list_old0)
                var_list_old0 = var_list
            elif flag == (1,):
                if len(var_list_old1):
                    assign_trainable_vars(sess, var_list, var_list_old1)
                    # print(var_list_old1)
                var_list_old1 = var_list
            else:
                if len(var_list_old2):
                    assign_trainable_vars(sess, var_list, var_list_old2)
                var_list_old2 = var_list
        else:
            print('')
            print(name + ' ' + 'fine tuning all ' + ','.join([v.name for v in var_all]))
        nmse_history = []
        for i in range(maxit + 1):
            if i % iv1 == 0:
                nmse = sess.run(nmse_, feed_dict={h_: hv, U: U1})  # validation results
                nmse = round(nmse, 5)
                if np.isnan(nmse):
                    raise RuntimeError('nmse is Nan')
                nmse_history = np.append(nmse_history, nmse)
                nmse_dB = 10 * np.log10(nmse)
                nmsebest_dB = 10 * np.log10(nmse_history.min())
                sys.stdout.write(
                    '\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i, nmse=nmse_dB, best=nmsebest_dB))
                sys.stdout.flush()
                if i % (iv1 * 100) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin() - 1
                    if age_of_best * iv1 > better_wait:
                        break
            rand_index = np.random.choice(trainingsize, size=100)
            h = ht[..., rand_index]
            h = np.transpose(h, [2, 0, 1])
            sess.run(train_, feed_dict={h_: h, U: U1})

        done = np.append(done, name)
        result_log = str('{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name, nmse=nmse_dB, i=i))
        log = np.append(log, result_log)
        # log = log + '\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name, nmse=nmse_dB, i=i)

        state['done'] = done
        state['log'] = log
        state['layernmse'] = layernmse

        save_trainable_vars(sess, savefile, snr=snr, **state)
    if nmse_dB is None:
        layernmse = layernmse
    else:
        layernmse = np.append(layernmse, nmse_dB)
    print('')
    print(layernmse)
    state['layernmse'] = layernmse
    save_trainable_vars(sess, savefile, snr=snr, **state)
    return sess
