# Some problems that make LP solvers sometimes fail

 - Add solvers to the file ``solvers.py``
 - Add problems to the file ``problems.py``
 - If data files are needed to define the problem, put them in the ``data`` directory
 - Add any methods / problems you want to skip to the corresponding skip list in ``main.py``

Note:

 - Not all solvers are exclusively LP solvers, just solvers that can solve LPs
 - For most of my problems I need the basic solution so simplex methods are preferred

Run:

```
python3 main.py
```

Requires:
 - ``numpy``
 - ``scipy``
 - ``cvxpy``
