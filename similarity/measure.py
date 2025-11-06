from functools import partial
from typing import Collection

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

no_nan = lambda x: jnp.nan_to_num(x, nan=0, posinf=0, neginf=0)

def measure(tens1: jnp.ndarray, tens2: jnp.ndarray, weight: jnp.ndarray) -> jax.Array:
    ##########
    # ORIGIN #
    ##########
    
    tot1_o = jnp.sum(tens1, 1)
    tens1n_ref_o = no_nan(tens1 / jnp.reshape(2 * tot1_o, (-1, 1)))
    tot2_o = jnp.sum(tens2, 1)
    tot_o = tot1_o + tot2_o
    weights_o = no_nan(tot_o / jnp.sum(tot1_o + tot2_o)) * weight
    
    tens1n_o = no_nan(tens1 / jnp.reshape(tot_o, (-1, 1)))
    tens2n_o = no_nan(tens2 / jnp.reshape(tot_o, (-1, 1)))
    tens2n_ref_o = no_nan(tens2 / jnp.reshape(2 * tot2_o, (-1, 1)))
    
    
    
    multstens12_o = tens1n_o - tens1n_ref_o
    multstens12_o = multstens12_o * multstens12_o
    
    multstens21_o = tens2n_o - tens1n_ref_o
    multstens21_o = multstens21_o * multstens21_o
    
    multstens12s_o = multstens12_o + multstens21_o
    cross_ent12_int_s_o = jnp.sum(multstens12s_o, 1)
    cross_ent12_o = jnp.sum(cross_ent12_int_s_o * weights_o)
    
    
    
    multstens12_2_o = tens1n_o - tens2n_ref_o
    multstens12_2_o = multstens12_2_o * multstens12_2_o
    
    multstens21_2_o = tens2n_o - tens2n_ref_o
    multstens21_2_o = multstens21_2_o * multstens21_2_o
    
    multstens12s_2_o = multstens12_2_o + multstens21_2_o
    cross_ent12_int_s_2_o = jnp.sum(multstens12s_2_o, 1)
    cross_ent21_o = jnp.sum(cross_ent12_int_s_2_o * weights_o)
    
    
    
    cross_ent_tot_o = cross_ent12_o * cross_ent21_o
    
    ###############
    # DESTINATION #
    ###############
    
    tot1_d = jnp.sum(tens1, 0)
    tens1n_ref_d = no_nan(tens1 / jnp.reshape(2 * tot1_d, (1, -1)))
    tot2_d = jnp.sum(tens2, 0)
    tot_d = tot1_d + tot2_d
    weights_d =no_nan(tot_d / jnp.sum(tot1_d + tot2_d)) * weight
    
    tens1n_d = no_nan(tens1 / jnp.reshape(tot_d, (1, -1)))
    tens2n_d = no_nan(tens2 / jnp.reshape(tot_d, (1, -1)))
    tens2n_ref_d = no_nan(tens2 / jnp.reshape(2 * tot2_d, (1, -1)))
    
    
    multstens12_d = tens1n_d - tens1n_ref_d
    multstens12_d = multstens12_d * multstens12_d
    
    multstens21_d = tens2n_d - tens1n_ref_d
    multstens21_d = multstens21_d * multstens21_d
    
    multstens12s_d = multstens12_d + multstens21_d
    cross_ent12_int_s_d = jnp.sum(multstens12s_d, 0)
    cross_ent12_d = jnp.sum(cross_ent12_int_s_d * weights_d)
    
    
    
    multstens12_2_d = tens1n_d - tens2n_ref_d
    multstens12_2_d = multstens12_2_d * multstens12_2_d
    
    multstens21_2_d = tens2n_d - tens2n_ref_d
    multstens21_2_d = multstens21_2_d * multstens21_2_d
    
    multstens12s_2_d = multstens12_2_d + multstens21_2_d
    cross_ent12_int_s_2_d = jnp.sum(multstens12s_2_d, 0)
    cross_ent21_d = jnp.sum(cross_ent12_int_s_2_d * weights_d)
    
    
    
    cross_ent_tot_d = cross_ent12_d * cross_ent21_d
    
    #########
    # TOTAL #
    #########
    
    return cross_ent_tot_o + cross_ent_tot_d


@partial(jax.jit, static_argnames=["n"])
def _levenshtein(a: ArrayLike, b: ArrayLike, ia: ArrayLike, ib: ArrayLike, n: int) -> jax.Array:
    xm2 = jnp.zeros([1])
    xm1 = jnp.asarray([a[0], b[0]])
    for i in range(1, n):
        x_true = jnp.empty([i + 2]).at[1:-1].set(
            xm2 + jnp.abs(a[(i-1)::-1] - b[:i])
        )
        x_false = jnp.minimum(
            jnp.full([i + 2], jnp.inf).at[ :-1].set(xm1 + a[i:     :-1]),
            jnp.full([i + 2], jnp.inf).at[1:  ].set(xm1 + b[ :(i+1)   ]),
        )
        i_cond = jnp.zeros([i + 2], dtype=bool).at[1:-1].set(ia[(i-1)::-1] == ib[:i])
        x = jax.lax.select(i_cond, x_true, x_false)
        xm2 = xm1
        xm1 = x
    
    y = x[0] + x[-1]
    
    x_true = xm2 + jnp.abs(a[(n-1)::-1] - b)
    x_false = jnp.minimum(
        xm1[1:  ] + a[(n-1)::-1],
        xm1[ :-1] + b,
    )
    i_cond = ia[(n-1)::-1] == ib
    x = jax.lax.select(i_cond, x_true, x_false)
    xm2 = xm1
    xm1 = x

    for i in range(1, n):
        x_true = xm2[1:-1] + jnp.abs(a[(n-1):(i-1):-1] - b[i:])
        x_false = jnp.minimum(
            xm1[1:  ] + a[(n-1):(i-1):-1],
            xm1[ :-1] + b[    i:        ],
        )
        i_cond = ia[(n-1):(i-1):-1] == ib[i:]
        x = jax.lax.select(i_cond, x_true, x_false)
        xm2 = xm1
        xm1 = x
    
    return jnp.where(y != 0, x / y, 0.0)


def nlod(tens1: ArrayLike, tens2: ArrayLike) -> jax.Array:
    n_area = tens1.shape[0]
    idx1 = jnp.argsort(tens1, axis=1, descending=True)
    idx2 = jnp.argsort(tens2, axis=1, descending=True)
    dist = jnp.zeros([1])
    
    def _loop_fn(i, d):
        idx_1o = idx1[i]
        idx_2o = idx2[i]
        tens1o_sorted = tens1[i, idx_1o]
        tens2o_sorted = tens2[i, idx_2o]
        return d + _levenshtein(tens1o_sorted, tens2o_sorted, idx_1o, idx_2o, n_area)
    
    dist = jax.lax.fori_loop(0, n_area, _loop_fn, dist)
    return dist / n_area


@jax.jit
def _ssim_l(
    x: ArrayLike,
    y: ArrayLike,
    c: float,
) -> jax.Array:
    mean_x = jnp.mean(x)
    mean_y = jnp.mean(y)
    return (2 * mean_x * mean_y + c) / (mean_x ** 2 + mean_y ** 2 + c)


@jax.jit
def _ssim_c(
    x: ArrayLike,
    y: ArrayLike,
    c: float,
) -> jax.Array:
    std_x = jnp.std(x)
    std_y = jnp.std(y)
    return (2 * std_x * std_y + c) / (std_x ** 2 + std_y ** 2 + c)


@jax.jit
def _ssim_str(
    x: ArrayLike,
    y: ArrayLike,
    c: float,
) -> jax.Array:
    std_x = jnp.std(x)
    std_y = jnp.std(y)
    cor_xy = jnp.mean(x * y) - jnp.mean(x) * jnp.mean(y)
    return (cor_xy + c) / (std_x * std_y + c)


@jax.jit
def _ssim(
    x: ArrayLike,
    y: ArrayLike,
    c1: float=1e-10,
    c2: float=1e-2,
    c3: float=5e-3,
    alpha: float=1.,
    beta: float=1.,
    gamma: float=1.,
) -> jax.Array:
    return _ssim_l(x, y, c1) ** alpha * _ssim_c(x, y, c2) ** beta * _ssim_str(x, y, c3) ** gamma


def gssi(tens1: ArrayLike, tens2: ArrayLike, clusters: Collection[ArrayLike]) -> jax.Array:
    tot_tens1 = jnp.sum(tens1)
    tot_tens2 = jnp.sum(tens2)
    sim = 0.
    n = 0
    for idx1 in clusters:
        for idx2 in clusters:
            subtens1 = tens1[idx1, :][:, idx2]
            subtens2 = tens2[idx1, :][:, idx2]
            ssim_val = _ssim(subtens1, subtens2)
            tot_subtens1 = jnp.sum(subtens1)
            tot_subtens2 = jnp.sum(subtens2)
            ssim_val = ssim_val * (tot_subtens1 + tot_subtens2) / (tot_tens1 + tot_tens2)
            sim = (n * sim + ssim_val) / (n + 1)
            n += 1
    return sim
