# main code from http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
import warnings
import numpy as np
import math
from numpy.linalg import eig, inv
from skimage.feature import canny
from numpy.linalg.linalg import LinAlgError

# TODO better split into two functions like belowx
# oder einfach nur die lezte fuktion 'mehrdimensional' machen
# TODO andere doppelfkt wieder killen

Epsilon = 1e-7


def get_contour_from_img(img, sigma=3, clip=True, clip_min=0., clip_max=0.4,
                         img_ord='channels_first'):
    img_ = img if img_ord == 'channels_first' else np.transpose(
        img, (2, 0, 1))
    if len(np.shape(img_)) == 3:
        img_ = np.sum(img_, 0) / 3.
    if len(np.shape(img_)) != 2:
        print np.shape(img_)
    if clip:
        img_ = np.clip(img_, clip_min, clip_max)
        img_ = (img_ - clip_min) / (clip_max - clip_min)
    return canny(img_, sigma)


def get_quad_distance(ellipse_par, x, y):
    s_hand = []
    cosTh = math.cos(-1. * ellipse_par[4])
    sinTh = math.sin(-1. * ellipse_par[4])
    for xi, yi in zip(x, y):
        s_hand.append((((((cosTh * (xi - ellipse_par[1]) - sinTh * (
            yi - ellipse_par[0])) / (ellipse_par[3] + Epsilon))**2) + (
            ((sinTh * (xi - ellipse_par[1]) + cosTh * (
                yi - ellipse_par[0])) / (ellipse_par[2] + Epsilon))
            ** 2)) - 1.)**2)
    return (np.sum(s_hand) / (len(s_hand) + Epsilon))**0.5

# todo better split into two functions like below


def points_from_input(input_, threshhold, pointskip=1):
    x = []
    y = []
    # print np.shape(input_)
    for i in xrange(0, len(input_)):
        for j in xrange(0, len(input_[i])):
            if input_[i][j] >= threshhold:
                x.append(i)
                y.append(j)
    if pointskip > 1:
        xn = []
        yn = []
        for i in xrange(0, len(x)):
            if not i % pointskip:
                xn.append(x[i])
                yn.append(y[i])
        x = xn
        y = yn
    return x, y


def fitEllipse(x, y):
    # print np.shape(x)
    # print np.shape(y)
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    try:
        E, V = eig(np.dot(inv(S), C))
    except LinAlgError, e:
        if np.shape(x)[0] and e != 'Singular matrix\n':
            print '\n'
            print repr(e)
            print '\n'
            raise LinAlgError(e)
        else:
            warnings.warn(
                'No entries in fitEllipse inputs, a is gonne be set to zeros')
            E = 0
            V = np.zeros((6, 1))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    # print a
    return a


def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c + Epsilon
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return -0.5 * np.arctan(2 * b / (a - c + Epsilon))


def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 +
                                                 4 * b * b / ((a - c) * (a - c) + Epsilon)) - (c + a)) + Epsilon
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 +
                                                 4 * b * b / ((a - c) * (a - c) + Epsilon)) - (c + a)) + Epsilon
    # if up< 0. or down1<=0. or down2<=0.: print up,down1,down2
    # epsilon=1e-10
    # if np.abs(up)<epsilon: up=0.
    # if np.abs(down1)<epsilon: down1=epsilon
    # if np.abs(down2)<epsilon: down2=epsilon
    res1 = np.sqrt(np.abs(up / down1))
    res2 = np.sqrt(np.abs(up / down2))
    return np.array([res1, res2])


def get_ellipse_par_from_a(a):
    return_val = np.asarray([((ellipse_center(a)[0])), (
        (ellipse_center(a)[1])), (
        (ellipse_axis_length(a)[0])), (
        (ellipse_axis_length(a)[1])),
        ellipse_angle_of_rotation(a)])
    return return_val


def get_ellipse_par(input_, pointskip=1):
    x, y = points_from_input(input_, threshhold=1., pointskip=pointskip)
    a = fitEllipse(np.asarray(x), np.asarray(y))
    return get_ellipse_par_from_a(a)


def _get_ellipse_kaggle_par(input_):
    # print np.shape(input_)
    x, y = points_from_input(get_contour_from_img(input_), threshhold=1.)
    try:
        a = fitEllipse(x, y)
        if len(a) > 9:
            warnings.warn(
                'from ellipse a was cut: %s, this warning shpuld not be raised anymore' % a[6:-1])
            a = a[0:6]
    except LinAlgError, e:
        print 'get_ellipse_kaggle_par'
        print np.shape(input_)
        raise LinAlgError(e)
    ax_len = ellipse_axis_length(a)
    ax_frac = ax_len[0] / (ax_len[1] + Epsilon)
    ax_frac_sqr = ax_frac**2
    quad_distance = get_quad_distance(get_ellipse_par_from_a(a), x, y)
    return_ = list(a) + [ax_frac, ax_frac_sqr, quad_distance]
    return np.array(return_)


def get_ellipse_kaggle_par(input_):
    if len(np.shape(input_)) <= 3:
        return _get_ellipse_kaggle_par(input_)
    else:
        return map(_get_ellipse_kaggle_par, input_)
