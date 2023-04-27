# **Linear Programming Solver with Simplex Algorithm and Branch-and-Bound**

This is a Python program that solves linear programming problems using the simplex algorithm with branch-and-bound to find integer values for x variables. The program takes a linear programming problem in standard form as input and outputs the optimal solution and optimal value.

### **Requirements**

- Python 3.10.x
- NumPy library

### **Run the program:**

```bash
$ python3.10 main.py examples/lp_glaces.txt
```

### **notes**

you can active the debug option from main.py:
the debug options will print intermediary tables.

```py
solver.debug(True)
```

### **Result**

```bash
-----entering LoadFromFile
Parsing  examples/lp_glaces.txt
Costs:  [-8. -9.]
AMatrix:  [[ 2.  5.]
 [50.  5.]
 [ 5. 50.]]
RHS:  [ 12. 150. 100.]
-----entering __initTableau
-----entering __isSolutionFeasible
-----entering __performPivots
Execution statistics
------------------------------------------------
Non-zero variables
[0 4 1]
x[0]     = 2.8750
x[1]     = 1.2500
z[2]     = 23.1250
------------------------------------------------
Num rows / cols     :  3  /  2
Tableau dimensions: :  (4, 6)
Optimiser status    :  OptStatus.Optimal
Objective Value     :  34.25
Nbr pivots Phase I  :  0
Nbr pivots Phase II :  3
------------------------------------------------
------------------------------------------------
Simplex tableau:  before PLNE  Opt status:  OptStatus.Optimal
------------------------------------------------
   1.000    0.000   -0.021    0.021    0.000    2.875
   0.000    0.000  -10.312    0.312    1.000   23.125
   0.000    1.000    0.208   -0.008    0.000    1.250
   0.000    0.000    1.708    0.092    0.000   34.250

------------- start PLNE
Integer solution found
   1.000    0.000    0.000    0.000    0.000   -1.000    3.000
   0.000    0.000    0.000  -10.000    1.000 -495.008   85.001
   0.000    1.000    0.000    0.200    0.000   10.000   -0.000
   0.000    0.000    1.000   -1.000    0.000  -48.001    6.000
   0.000    0.000    0.000    1.800    0.000   82.001   24.000

OBJECTIVE VALUE : 24.00

Integer solution found
   1.000    0.000    0.000    0.000    0.000    1.000    0.000    2.000
   0.000    0.000    0.000    0.000    1.000   -4.999  -50.000   39.999
   0.000    1.000    0.000    0.000    0.000    0.000    1.000    1.000
   0.000    0.000    0.000    1.000    0.000  -50.001   -5.000   45.001
   0.000    0.000    1.000    0.000    0.000   -2.000   -5.000    3.000
   0.000    0.000    0.000    0.000    0.000    8.000    9.000   25.000

OBJECTIVE VALUE : 25.00

Integer solution found
   1.000    0.000    0.000    0.000    0.200    0.000   10.001   -0.000
   0.000    0.000    1.000    0.000   -0.400    0.000  -15.002    2.001
   0.000    1.000    0.000    0.000    0.000    0.000   -1.000    2.000
   0.000    0.000    0.000    1.000  -10.002    0.000 -495.081  140.023
   0.000    0.000    0.000    0.000   -0.200    1.000  -10.001    2.000
   0.000    0.000    0.000    0.000    1.600    0.000   71.012   17.997

OBJECTIVE VALUE : 18.00


Best Integer solution
   1.000    0.000    0.000    0.000    0.000    1.000    0.000    2.000
   0.000    0.000    0.000    0.000    1.000   -4.999  -50.000   39.999
   0.000    1.000    0.000    0.000    0.000    0.000    1.000    1.000
   0.000    0.000    0.000    1.000    0.000  -50.001   -5.000   45.001
   0.000    0.000    1.000    0.000    0.000   -2.000   -5.000    3.000
   0.000    0.000    0.000    0.000    0.000    8.000    9.000   25.000

OBJECTIVE VALUE : 25.00

integer solutions: [24.0, 25.0, 18.0]
execution time: 0.004 sec

------------- finish PLNE
```
