import tensorflow as tf
import math
import numpy as np

pi = math.pi


def shrink_soft_threshold(r, rvar, theta):
    if len(theta.get_shape()) > 0 and theta.get_shape != (1,):
        lam = theta[0] * tf.sqrt(rvar)
        scale = theta[1]
    else:
        lam = theta * tf.sqrt(rvar)
        scale = None

    lam = tf.abs(lam)
    lam = tf.maximum(lam, 0)
    phase = tf.angle(r)
    xhat = tf.complex(tf.maximum(tf.abs(r) - lam, 0) * tf.cos(phase), tf.maximum(tf.abs(r) - lam, 0) * tf.sin(phase))
    # dxdr = tf.reduce_mean(
    #     tf.cast((tf.abs(r) - lam) > 0, tf.complex128) * tf.cast((1 - lam / (2 * tf.abs(r))), tf.complex128), 0)
    dxdr = tf.reduce_mean(tf.cast((tf.abs(r) - lam) > 0, tf.complex128), 0)
    if scale is not None:
        xhat = xhat
        dxdr = dxdr
    return xhat, dxdr


def shrink_MMV(r, x_gain, tau, L):
    G = int(r.shape[1])
    K = int(r.shape[2])
    lam = L / G
    delta = 1 / tau - 1 / (tau + 1)
    filter_gain = tf.cast(
        1 / (1 + tau) / (1 + (1 - lam) / lam * tf.exp(K * (tf.log(1 + 1 / tau) - delta * x_gain))),
        tf.complex128)
    a = tf.matrix_diag(filter_gain)
    # b = tf.matrix_diag(tf.transpose(filter_gain, (1, 0)))
    xhat = tf.matmul(a, r)
    # xhat = tf.transpose(xhat, (1, 2, 0))
    return xhat, filter_gain


def shrink_bgest(r, rvar, K, theta):
    xvar1 = abs(theta[0, ...])
    loglam = theta[1, ...]  # log(1/lambda - 1)
    beta = 1 / (1 + rvar / xvar1)
    r_gain = tf.reduce_sum(tf.square(tf.abs(r)), axis=2)
    r2scale = r_gain * beta / rvar
    rho = tf.exp(loglam + .5 * K * tf.log(1 + xvar1 / rvar) - .5 * r2scale)
    rho1 = rho + 1
    gain = tf.cast(beta / rho1, tf.complex128)
    xhat = tf.matmul(tf.matrix_diag(gain), r)
    # dxdr = beta * ((1 + rho * (1 + r2scale)) / tf.square(rho1))
    dxdr = tf.reduce_mean(gain, axis=1)
    return (xhat, dxdr)


def get_shrinkage_function(name):
    try:
        return {
            'bg': (shrink_bgest, [[1.0], [math.log(1 / .1 - 1)]]),
            'MMV': (shrink_MMV, [1.0]),
            'soft': (shrink_soft_threshold, [[1.0], [1.0]]),
        }[name]
    except KeyError as ke:
        raise ValueError('unrecognized shrink function %s' % name)
        sys.exit(1)
