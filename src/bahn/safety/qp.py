"""Exact batched solver for the two-variable QP the safety filter needs.

    minimise  sum_j w_j (u_j - u_nom_j)^2      subject to  A u >= b

With only two decision variables the solution can be enumerated exactly instead
of iterated: the optimum of a strictly convex quadratic over a polyhedron lies
either at the unconstrained point, on one active constraint, or at the
intersection of two.  Enumerating all three cases and taking the feasible
candidate with the smallest objective is exact, has no tolerances to tune, no
iteration limit to hit, and no dependency to install -- and it vectorises over
the whole batch, which an off-the-shelf solver called in a Python loop does not.

``tests/test_qp.py`` checks it against ``scipy.optimize.minimize`` on random
instances, including degenerate and infeasible ones.
"""

from __future__ import annotations

import numpy as np

FEAS_TOL = 1e-9


def solve_qp2(
    u_nom: np.ndarray,
    weights: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a batch of 2-variable inequality-constrained least-squares problems.

    Args:
        u_nom: ``(n, 2)`` point to project.
        weights: ``(2,)`` strictly positive weights on each coordinate.
        A: ``(n, m, 2)`` constraint normals.
        b: ``(n, m)`` constraint offsets; the feasible set is ``A u >= b``.

    Returns:
        ``(u, feasible)`` with ``u`` of shape ``(n, 2)`` and ``feasible`` a
        boolean ``(n,)``.  Where a problem is infeasible ``u`` is ``u_nom`` and
        the caller is expected to substitute its own fallback.
    """
    u_nom = np.asarray(u_nom, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m, _ = A.shape
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive")

    # Whiten so the objective becomes plain squared distance: z = sqrt(w)*(u-u_nom).
    root = np.sqrt(w)
    Ab = A / root[None, None, :]
    c = b - np.einsum("nmj,nj->nm", A, u_nom)

    cands = [np.zeros((n, 1, 2))]

    # One active constraint: the projection of the origin onto a_j.z = c_j.
    norm2 = np.einsum("nmj,nmj->nm", Ab, Ab)
    safe = np.maximum(norm2, 1e-18)
    single = Ab * (c / safe)[..., None]
    single = np.where((norm2 > 1e-18)[..., None], single, np.inf)
    cands.append(single)

    # Two active constraints: the vertex where a_j.z = c_j and a_k.z = c_k.
    if m >= 2:
        jj, kk = np.triu_indices(m, k=1)
        aj, ak = Ab[:, jj, :], Ab[:, kk, :]
        cj, ck = c[:, jj], c[:, kk]
        det = aj[..., 0] * ak[..., 1] - aj[..., 1] * ak[..., 0]
        ok = np.abs(det) > 1e-12
        d = np.where(ok, det, 1.0)
        zx = (cj * ak[..., 1] - ck * aj[..., 1]) / d
        zy = (aj[..., 0] * ck - ak[..., 0] * cj) / d
        pair = np.stack([zx, zy], axis=-1)
        cands.append(np.where(ok[..., None], pair, np.inf))

    z = np.concatenate(cands, axis=1)  # (n, q, 2)
    finite = np.isfinite(z).all(axis=-1)
    z = np.where(finite[..., None], z, 0.0)

    residual = np.einsum("nmj,nqj->nqm", Ab, z) - c[:, None, :]
    feasible = (residual >= -FEAS_TOL).all(axis=-1) & finite

    cost = np.einsum("nqj,nqj->nq", z, z)
    cost = np.where(feasible, cost, np.inf)
    best = cost.argmin(axis=1)
    rows = np.arange(n)
    z_best = z[rows, best]
    any_feasible = np.isfinite(cost[rows, best])

    u = u_nom + z_best / root[None, :]
    return np.where(any_feasible[:, None], u, u_nom), any_feasible
