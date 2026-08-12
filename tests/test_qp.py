"""The 2-variable QP solver, checked against a general-purpose solver.

Enumerating active sets is only worth doing if it is genuinely exact, so these
compare against ``scipy.optimize.minimize`` on random instances, including
degenerate rows and infeasible systems.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from bahn.safety.qp import solve_qp2


def scipy_reference(u_nom, w, A, b, restarts=6, seed=0):
    """Best feasible point scipy can find, or None if it finds none."""
    rng = np.random.default_rng(seed)
    f = lambda z: float((w * (z - u_nom) ** 2).sum())  # noqa: E731
    cons = [{"type": "ineq", "fun": (lambda z, i=i: float(A[i] @ z - b[i]))} for i in range(len(b))]
    best = None
    for _ in range(restarts):
        r = minimize(f, u_nom + rng.normal(scale=2.0, size=2), constraints=cons, method="SLSQP")
        if r.success and all(A[i] @ r.x - b[i] > -1e-9 for i in range(len(b))):
            best = r.fun if best is None else min(best, r.fun)
    return best


def test_unconstrained_optimum_is_returned_untouched():
    u_nom = np.array([[0.4, -0.2]])
    A = np.array([[[1.0, 0.0]]])
    b = np.array([[-5.0]])  # slack constraint
    u, ok = solve_qp2(u_nom, np.array([1.0, 1.0]), A, b)
    assert ok[0]
    assert u[0] == pytest.approx(u_nom[0])


def test_single_active_constraint_is_an_orthogonal_projection():
    u_nom = np.array([[0.0, 0.0]])
    A = np.array([[[1.0, 0.0]]])
    b = np.array([[0.5]])  # requires u0 >= 0.5
    u, ok = solve_qp2(u_nom, np.array([1.0, 1.0]), A, b)
    assert ok[0]
    assert u[0] == pytest.approx([0.5, 0.0])


def test_weights_tilt_the_projection():
    """A heavily weighted axis should move less than a lightly weighted one."""
    u_nom = np.array([[0.0, 0.0]])
    A = np.array([[[1.0, 1.0]]])
    b = np.array([[1.0]])
    heavy_v, _ = solve_qp2(u_nom, np.array([100.0, 1.0]), A, b)
    heavy_w, _ = solve_qp2(u_nom, np.array([1.0, 100.0]), A, b)
    assert abs(heavy_v[0, 0]) < abs(heavy_v[0, 1])
    assert abs(heavy_w[0, 1]) < abs(heavy_w[0, 0])


def test_infeasible_system_is_reported_not_guessed():
    u_nom = np.array([[0.0, 0.0]])
    A = np.array([[[1.0, 0.0], [-1.0, 0.0]]])
    b = np.array([[1.0, 1.0]])  # u0 >= 1 and u0 <= -1
    _, ok = solve_qp2(u_nom, np.array([1.0, 1.0]), A, b)
    assert not ok[0]


def test_degenerate_zero_row_does_not_crash():
    u_nom = np.array([[0.3, 0.3]])
    A = np.array([[[0.0, 0.0], [1.0, 0.0]]])
    b = np.array([[-1.0, 0.5]])
    u, ok = solve_qp2(u_nom, np.array([1.0, 1.0]), A, b)
    assert ok[0]
    assert u[0, 0] == pytest.approx(0.5)


def test_batch_is_independent_across_rows():
    u_nom = np.array([[0.0, 0.0], [0.0, 0.0]])
    A = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    b = np.array([[0.5], [0.25]])
    u, ok = solve_qp2(u_nom, np.array([1.0, 1.0]), A, b)
    assert ok.all()
    assert u[0] == pytest.approx([0.5, 0.0])
    assert u[1] == pytest.approx([0.0, 0.25])


@pytest.mark.parametrize("seed", range(12))
def test_matches_scipy_on_random_instances(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 8))
    u_nom = rng.normal(size=(1, 2))
    w = rng.uniform(0.3, 4.0, size=2)
    A = rng.normal(size=(1, m, 2))
    b = rng.normal(size=(1, m)) * 0.8

    u, ok = solve_qp2(u_nom, w, A, b)
    ref = scipy_reference(u_nom[0], w, A[0], b[0], seed=seed)

    if not ok[0]:
        assert ref is None, "declared infeasible but scipy found a feasible point"
        return
    assert (A[0] @ u[0] - b[0] >= -1e-8).all(), "returned an infeasible point"
    if ref is not None:
        mine = float((w * (u[0] - u_nom[0]) ** 2).sum())
        assert mine <= ref + 1e-6 * max(abs(ref), 1.0)
