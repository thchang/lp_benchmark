import solvers
import problems

# Skip the following to save time
skip_solvers = ["osqp_primal", "osqp_dual", "scs_primal", "scs_dual", "ecos_dual"]
skip_problems = ["KddDelaunay"]

# Loop over all solvers and problems and print success rates
for keyi in solvers.__dict__:
    solveri = solvers.__dict__[keyi]
    if callable(solveri) and keyi not in skip_solvers:
        print()
        print("="*60)
        print(keyi)
        print("="*60)
        for keyj in problems.__dict__:
            problemj = problems.__dict__[keyj]
            if callable(problemj) and keyj not in skip_problems:
                print()
                print(keyj, end=" ")
                print(f"% solved: {problemj(solveri)*100}")
print()
print("Done")
