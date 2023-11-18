""" Problems to test.

Implement a new problem here.

Each problems should take a function as input, with signature

``solver(A, b, c) -> bool``

which solves the LP :

min_x <c, x> s.t. Ax <= b (or its dual)

and returns True iff the problem was solved successfully.

"""


def SyntheticDelaunay(solver):
    """ Check whether solvers can handle a tall dense LP.

    Solves the LP formulation in Section 2.1 of ACM TOMS Alg. 1012
    (not for the basic solution, just the optimal value) for many synthetic
    ML problems.

    Args:
        solver (function): The LP solver matching the interface described
            above.

    Returns:
        float: Proportion of successful solves over all problems

    """

    import numpy as np
    from scipy.stats import qmc

    nfail = 0
    iseed = 0
    # Try 16 total training sets + sizes simulating various ML training sets
    for n in [1000, 5000]:                            # number of pts
        for d in [10, 25]:                            # dimension
            for kappa in [10.0, 1000.0]:              # conditioning
                for sampler in ["halton", "lhs"]: # sampling methods
                    # Average over 5 random seeds
                    for si in range(5):
                        np.random.seed(iseed)
                        iseed += 1
                        # Create sythetic ML training set + 10 test points
                        if sampler == "sobol":
                            train = qmc.Sobol(d, scramble=True).random(n)
                        elif sampler == "uniform":
                            train = np.random.random_sample((n, d))
                        elif sampler == "lhs":
                            train = qmc.LatinHypercube(d, scramble=True).random(n)
                        elif sampler == "halton":
                            train = qmc.Halton(d, scramble=True).random(n)
                        sigmas = np.linspace(1.0, kappa, num=d)
                        for i in range(n):
                            train[i, :] = (train[i,:] - 0.5) / sigmas + 0.5
                        test = np.random.random_sample((10, d))
                        # Allocate memory for problem
                        A = np.ones((train.shape[0], train.shape[1] + 1))
                        b = np.ones(train.shape[0])
                        c = np.ones(test.shape[1] + 1)
                        for qi in test:
                            # Create A, b, and c
                            for i, xi in enumerate(train):
                                A[i, :-1] = -xi[:]
                                A[i, -1] = 1.0
                                b[i] = np.linalg.norm(xi) ** 2
                            c[:-1] = -qi[:]
                            # Try solver and count failures
                            if solver(A, b, c):
                                nfail += 1
    return nfail / 800

def KddDelaunay(solver):
    """ Check whether solvers can handle a tall dense LP.

    Solves the LP formulation in Section 2.1 of ACM TOMS Alg. 1012
    (not for the basic solution, just the optimal value) on a
    real-world ML data set (KDD Cup 99, preprocessed as described
    in the ACM TOMS Remark on Alg. 1012).

    Args:
        solver (function): The LP solver matching the interface described
            above.

    Returns:
        float: Proportion of successful solves over various train/test
        splits and subsamples of the full data set

    """

    import csv
    import numpy as np

    fname = "data/KDDCUP99_clean.dat"

    # Load the data set for testing
    with open(fname, "r") as fp:
        csv_reader = csv.reader(fp)
        for i, rowi in enumerate(csv_reader):
            if i == 0:
                pass
            elif i == 1:
                d = int(rowi[0])
                n = int(rowi[1])
                m = int(rowi[2])
                train = np.zeros((n, d))
                test = np.zeros((m, d))
            elif i - 2 < n:
                for j, colj in enumerate(rowi):
                    train[i-2, j] = float(colj.strip())
            else:
                for j, colj in enumerate(rowi):
                    test[i-2-n, j] = float(colj.strip())

    nfail = 0
    # Generate various training sizes
    for nk in [1000, 10000, 15000]:
        # Subsample the data
        mk = 20
        pts = np.zeros((nk, d))
        q = np.zeros((mk, d))
        # Use 10 different random seeds
        for j in range(10):
            # Randomly subsample nk train pts and mk test pts
            np.random.seed(j)
            itrain = np.random.choice(n, nk, replace=False)
            itest = np.random.choice(m, mk, replace=False)
            pts[:, :] = train[itrain,:]
            q[:, :] = test[itest,:]
            # Allocate memory for problem
            A = np.ones((pts.shape[0], pts.shape[1] + 1))
            b = np.ones(pts.shape[0])
            c = np.ones(test.shape[1] + 1)
            # Loop over all test points
            for qi in q:
                # Create A, b, and c
                for i, xi in enumerate(pts):
                    A[i, :-1] = -xi[:]
                    A[i, -1] = 1.0
                    b[i] = np.linalg.norm(xi) ** 2
                c[:-1] = -qi[:]
                # Try solver and count failures
                if solver(A, b, c):
                    nfail += 1
    return nfail / 600
