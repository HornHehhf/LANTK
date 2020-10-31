import numpy as np
from FWHT import FWHT
from scipy import sparse
import numpy.random as npr
import math


def approx_bound(eps, n):
    return int(2 / eps ** 2 * math.log(n) + 1.0)


def fast_sample(n, sample_size):
    swap_records = {}
    sample_wor = np.empty(sample_size, dtype=int)
    for i in range(sample_size):
        rand_ix = npr.randint(i, n)

        if i in swap_records:
            el1 = swap_records[i]
        else:
            el1 = i

        if rand_ix in swap_records:
            el2 = swap_records[rand_ix]
        else:
            el2 = rand_ix

        swap_records[rand_ix] = el1
        sample_wor[i] = el2
        if i in swap_records:
            del swap_records[i]
    return sample_wor


def nextPow(d_act):
    d_act = d_act - 1
    d_act |= d_act >> 1
    d_act |= d_act >> 2
    d_act |= d_act >> 4
    d_act |= d_act >> 8
    d_act |= d_act >> 16
    d_act += 1
    return d_act


def fjlt(A, k, q):
    (d, n) = A.shape
    # Calculate the next power of 2
    d_act = nextPow(d)
    sc_ft = np.sqrt(d_act / float(d * k))
    # Calculate D plus some constansts
    D = npr.randint(0, 2, size=(d, 1)) * 2 * sc_ft - sc_ft
    DA = np.zeros((d_act, n))
    DA[0:d, :] = A * D
    print('get da done', DA.shape)

    # Apply hadamard transform to each row
    hda = np.apply_along_axis(FWHT, 0, DA).squeeze()
    print('get hda done', hda.shape)

    # Apply P transform
    sample_size = npr.binomial(k * d, q)
    indc = fast_sample(k * d, sample_size)
    p_rows, p_cols = np.unravel_index(indc, (k, d))
    p_data = npr.normal(loc=0, scale=math.sqrt(1 / q), size=len(p_rows))
    P = sparse.csr_matrix((p_data, (p_rows, p_cols)), shape=(k, d_act))
    print('get P done', (k, d_act))
    return P.dot(hda)


def fjlt_usp(A, k):
    (d, n) = A.shape
    # Calculate the next power of 2
    d_act = nextPow(d)
    sc_ft = np.sqrt(d_act / float(d * k))
    # Calculate D plus some constansts
    D = npr.randint(0, 2, size=(d, 1)) * 2 * sc_ft - sc_ft
    DA = np.zeros((d_act, n))
    DA[0:d, :] = A * D

    # Apply hadamard transform to each row
    hda = np.apply_along_axis(FWHT, 0, DA)

    # Apply P transform
    p_cols = fast_sample(d, k)
    p_rows = np.array(range(k))
    p_data = npr.randint(0, 2, size=k) * 2 - 1
    P = sparse.csr_matrix((p_data, (p_rows, p_cols)), shape=(k, d_act))
    return P.dot(hda)
