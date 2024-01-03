# %%
import numpy as np
from tqdm import tqdm

# %%
# Given: matrices A, B, C, D each nxn

def cov_estimate(A, B, C, D):
    """<pi_A, pi_C><pi_A, pi_D><pi_B, pi_C><pi_B, pi_D> / <pi_A><pi_B><pi_C><pi_D>"""
    ans = 1
    ans *= A.sum(axis=0).dot(C.sum(axis=1))
    ans *= C.sum(axis=0).dot(B.sum(axis=1))
    ans *= B.sum(axis=0).dot(D.sum(axis=1))
    ans *= D.sum(axis=0).dot(A.sum(axis=1))

    ans /= A.sum()
    ans /= B.sum()
    ans /= C.sum()
    ans /= D.sum()
    return ans

def var(A, B):
    """<pi_A, pi_B> / <pi_A><pi_B>"""
    return (A**2).sum() * (B**2).sum()

def rand_boolean_matrix(n):
    rt = np.random.randint(0, 2, (n, n)).astype(float)

    while rt.sum() == 0:
       rt = rand_boolean_matrix(n)
    return rt

def rand_pos_matrix(n):
    return abs(np.random.rand(n, n))

def stunted_rpm(n):
    rt = rand_pos_matrix(n)
    rt[2,:] *= 0
    rt[:,2] *= 0
    return rt
# %%
n = 2
trials = 10000000
func = rand_pos_matrix
for i in tqdm(list(range(trials))):
    a = func(n)
    b = func(n)
    c = func(n)
    d = func(n)

    ce = cov_estimate(a, b, c, d)
    vab = var(a, b)
    vcd = var(c, d)
    if ce > (vab * vcd)**0.5:
        print(f"n = {n}")
        print(f"a={a}")
        print(f"b={b}")
        print(f"c={c}")
        print(f"d={d}")
        print(f"cov = {ce}")
        print(f"var_ab = {vab}")
        print(f"var_cd = {vcd}")
        print("not psd because cov > sqrt(var_ab * var_cd)")
        break
# %%
# Hand-crafted boolean example showing that this estimate doesn't give PSD matrices, even in the boolean case
a = np.zeros((4, 4))
a[0,:] = 1 
a[:,0] = 1 
ce = cov_estimate(a, a, a, a)
vab = var(a, a)
print(f"cov = {ce}")
print(f"var_ab = {vab}")