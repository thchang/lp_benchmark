""" Solvers to test.

Each solver accepts inputs A, b, and c and returns True
iff the problem was solved (otherwise False).

Must solve a LP in the following form (or its dual):

min_x <c, x> s.t. Ax <= b

Signature should be:

solver(A, b, c): --> bool

"""


def ecos_primal(A, b, c):
    """ Solves min <c, x> s.t. Ax <= b (primal). """

    import cvxpy as cp

    x = cp.Variable(c.size)
    prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
    try:
        prob.solve(solver="ECOS")
        return True
    except BaseException:
        return False

def ecos_dual(A, b, c):
    """ Solves min <c, x> s.t. Ax <= b (dual). """

    import cvxpy as cp

    y = cp.Variable(b.size)
    prob = cp.Problem(cp.Minimize(b.T @ y), [A.T @ y == c, y >= 0])
    try:
        prob.solve(solver="ECOS")
        return True
    except BaseException:
        return False

def osqp_primal(A, b, c):
    """ Solves min <c, x> s.t. Ax <= b (primal). """

    import cvxpy as cp

    x = cp.Variable(c.size)
    prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
    try:
        prob.solve(solver="OSQP")
        return True
    except BaseException:
        return False

def osqp_dual(A, b, c):
    """ Solves min <c, x> s.t. Ax <= b (dual). """

    import cvxpy as cp

    y = cp.Variable(b.size)
    prob = cp.Problem(cp.Minimize(b.T @ y), [A.T @ y == c, y >= 0])
    try:
        prob.solve(solver="OSQP")
        return True
    except BaseException:
        return False

def scs_primal(A, b, c):
    """ Solves min <c, x> s.t. Ax <= b (primal). """

    import cvxpy as cp

    x = cp.Variable(c.size)
    prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
    try:
        prob.solve(solver="SCS")
        return True
    except BaseException:
        return False

def scs_dual(A, b, c):
    """ Solves min <c, x> s.t. Ax <= b (dual). """

    import cvxpy as cp

    y = cp.Variable(b.size)
    prob = cp.Problem(cp.Minimize(b.T @ y), [A.T @ y == c, y >= 0])
    try:
        prob.solve(solver="SCS")
        return True
    except BaseException:
        return False

def highs_ds_primal(A, b, c):
    """ Solves min <c, x> s.t. Ax <= b (primal). """

    import cvxpy as cp

    x = cp.Variable(c.size)
    prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
    try:
        prob.solve(solver="SCIPY", scipy_options={"method": "highs-ds"})
        return True
    except BaseException:
        return False

def highs_ds_dual(A, b, c):
    """ Solves min <c, x> s.t. Ax <= b (dual). """

    import cvxpy as cp

    y = cp.Variable(b.size)
    prob = cp.Problem(cp.Minimize(b.T @ y), [A.T @ y == c, y >= 0])
    try:
        prob.solve(solver="SCIPY", scipy_options={"method": "highs-ds"})
        return True
    except BaseException:
        return False

def highs_ipm_primal(A, b, c):
    """ Solves min <c, x> s.t. Ax <= b (primal). """

    import cvxpy as cp

    x = cp.Variable(c.size)
    prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
    try:
        prob.solve(solver="SCIPY", scipy_options={"method": "highs-ipm"})
        return True
    except BaseException:
        return False

def highs_ipm_dual(A, b, c):
    """ Solves min <c, x> s.t. Ax <= b (dual). """

    import cvxpy as cp

    y = cp.Variable(b.size)
    prob = cp.Problem(cp.Minimize(b.T @ y), [A.T @ y == c, y >= 0])
    try:
        prob.solve(solver="SCIPY", scipy_options={"method": "highs-ipm"})
        return True
    except BaseException:
        return False
