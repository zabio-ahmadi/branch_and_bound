import numpy as np
from os.path import isfile
import time, sys, math
from enum import Enum
from copy import deepcopy

from src.simplexe import *
from src.constants import *

from math import floor, ceil
from typing import *

from numba import njit

import numpy.typing as npt

TableauArray = np.dtype([('name', 'U10'), ('value', float)], align=True)

class BranchAndBound(Simplexe):
    def __init__(self):
        super().__init__()
        self.index = []
        self.depth = 0

    def go(self, lpFileName: str, printDetails: bool = False) -> None:
        self._Simplexe__PrintDetails = printDetails
        self.LoadFromFile(lpFileName, printDetails) # loads file and solves it with the simplex method
        #self = self.round_numpy_array(self)
        self.PrintTableau("before PLNE")
        self.PLNE()
        print("------------- finish PLNE")
        #self.PrintSolution()

    def create_bounds(self, tableau: 'BranchAndBound', isLeft=True):
        whichRow = tableau.index[tableau.depth % len(tableau.index)]
        whichVariable = tableau.depth
        rhs_val = tableau._Simplexe__tableau[whichRow][-1]
        abs_val = floor(rhs_val) if isLeft else -1 * ceil(rhs_val)
        new_line = [0] * (len(tableau._Simplexe__tableau[0]) - 1) + [1] + [abs_val]
        sign = 1 if isLeft else -1
        new_line[whichVariable] = sign

        tableau._Simplexe__tableau = np.hstack((tableau._Simplexe__tableau[:, :-1], np.atleast_2d([0] * len(tableau._Simplexe__tableau)).T, tableau._Simplexe__tableau[:, -1:]))

        tableau._Simplexe__tableau = np.vstack((tableau._Simplexe__tableau[:-1], new_line, tableau._Simplexe__tableau[-1:]))

        if isLeft:
            tableau._Simplexe__tableau[-2] -= tableau._Simplexe__tableau[whichRow]
        else:
            tableau._Simplexe__tableau[-2] += tableau._Simplexe__tableau[whichRow]
        tableau.NumCols += 1
        tableau.NumRows += 1
        tableau._Simplexe__basicVariables = np.append(tableau._Simplexe__basicVariables, np.max(tableau._Simplexe__basicVariables)+1)
        tableau._Simplexe__basicVariables = np.where(tableau._Simplexe__basicVariables >= tableau.NumCols, tableau._Simplexe__basicVariables + 1, tableau._Simplexe__basicVariables)
        
        #tableau.LP.AMatrix = np.vstack((tableau.LP.AMatrix, np.array([sign * (whichVariable==0), sign * (whichVariable==1)])))
        #tableau.LP.RHS = np.append(tableau.LP.RHS, abs_val)

        #input_string = self.generate_input_string(tableau, new_line)
        return tableau

    def create_bounds2(self, tableau: 'BranchAndBound', isLeft: bool = True) -> 'BranchAndBound':
        whichRow = tableau.index[tableau.depth % len(tableau.index)]
        whichVariable = tableau.depth
        rhs_val = tableau._Simplexe__tableau[whichRow][-1]
        sign = 1 if isLeft else -1
        abs_val = floor(rhs_val) if isLeft else ceil(rhs_val)
        tableau.NumRows += 1
        AMatrix_row = np.zeros(tableau.LP.AMatrix.shape[1])
        AMatrix_row[whichVariable] = sign
        tableau.LP.AMatrix = np.vstack((tableau.LP.AMatrix, AMatrix_row))
        tableau.LP.RHS = np.append(tableau.LP.RHS, abs_val * sign)

        return tableau

    def PLNE(self):
        def is_almost_integer(num: float, threshold: float = 0.01) -> bool:
            return abs(num - round(num)) < threshold


        def update_nodes(node: 'BranchAndBound', list_node: List) -> List:
            node.index = [i for i in range(len(node._Simplexe__basicVariables)) if node._Simplexe__basicVariables[i] < node.NumCols]

            left_tableau = self.create_bounds(deepcopy(node), True)
            left_tableau.depth += 1
            right_tableau = self.create_bounds(deepcopy(node), False)
            right_tableau.depth += 1
            list_node.append(left_tableau)
            list_node.append(right_tableau)

            node.index = []
            return list_node

        #numRows, numCols = self._Simplexe__tableau.shape

        objval_max = self.ObjValue()

        temp_node = deepcopy(self)

        list_node = []
        list_sol = []
        z_PLNE = float('-inf')


        iteration = 0
        max_iterations = 1000

        list_node = update_nodes(temp_node, list_node)
        best_tableau = None


        while list_node and iteration < max_iterations:
            iteration += 1
            node = list_node.pop(0)
            print("depth",node.depth)

            #node = self.round_numpy_array(node)
            #node = self.reorder(node)
            node.PrintTableau("before solve")
            #node.OptStatus == OptStatus.Unknown
            #node.initPhaseI(deepcopy(node))
            #node._Simplexe__solvePhaseI()
            #node._Simplexe__solveProblem(OptStatus.Unknown)
            node = self.solve_tableau(node)
            node.PrintTableau("after solve")

            objval = node.ObjValue()
            print("objVal after pivots", objval)

            if objval < z_PLNE or objval > objval_max: # no need to check anything if current objective value is worse than best
                continue

            isInteger = is_almost_integer(objval) and not any(t < -1e-6 for t in node._Simplexe__tableau[:-1, -1])
            isInteger = is_almost_integer(objval)
            if isInteger:
                if objval > z_PLNE:
                    z_PLNE = objval
                    best_tableau = node
                    list_sol.append(node)
                    print("Better solution found")
                    node.PrintTableau("Branch and Bound Solution")
                    print("OBJECTIVE VALUE : {:.2f}".format(z_PLNE))
                #sys.exit(0)
            elif all(t >= 0 for t in node._Simplexe__tableau[:-1, -1]):  # Update nodes only when the current node has a feasible solution
                list_node = update_nodes(node, list_node)

        # Print the best solution found after searching all nodes
        if best_tableau:
            print("\nBest solution found")
            best_tableau.PrintTableau("Best BnB Solution")
            print("OBJECTIVE VALUE : {:.2f}".format(z_PLNE))
    
    def reorder(self, tableau):
        basics = tableau._Simplexe__basicVariables
        basics = np.where(basics >= tableau.NumCols, basics - tableau.NumCols, basics)
        basics = np.append(basics, np.max(basics)+1)
        tableau._Simplexe__tableau = np.take(tableau._Simplexe__tableau, basics, axis=0)

        # reorder the index array so that it goes from 0 to max
        tableau._Simplexe__basicVariables = np.sort(tableau._Simplexe__basicVariables)
        return tableau

    def generate_input_string(self, tableau, new_line):
        numRows, numCols = tableau._Simplexe__tableau.shape

        # Generate objective function string
        obj_str = "max;" + ";".join(map(str, tableau.LP.Costs)) + ";"

        # Generate constraint strings
        constraint_strs = []
        for i in range(tableau.LP.AMatrix.shape[0]):
            constraint_str = ";".join(map(str, tableau.LP.AMatrix[i])) + ";<=" + ";" + str(tableau.LP.RHS[i]) + ";"
            constraint_strs.append(constraint_str)

        # Add the new constraint
        new_constraint_str = ";".join(map(str, new_line[:-1])) + ";<=" + ";" + str(new_line[-1]) + ";"
        constraint_strs.append(new_constraint_str)

        # Combine all parts of the input string
        input_string = obj_str + "\n" + "\n".join(constraint_strs) + "\n"

        return input_string

    def round_numpy_array(self, arr: 'BranchAndBound', decimals: int = 6) -> 'BranchAndBound':
        # convert the array to a floating-point type
        arr._Simplexe__tableau = arr._Simplexe__tableau.astype(float)

        # round the array to a maximum of 6 decimal places
        arr._Simplexe__tableau = np.round(arr._Simplexe__tableau, decimals=decimals)

        # convert any non-float values back to strings
        for i in range(arr._Simplexe__tableau.shape[0]):
            for j in range(arr._Simplexe__tableau.shape[1]):
                if not isinstance(arr._Simplexe__tableau[i, j], float):
                    arr[i, j] = str(arr._Simplexe__tableau[i, j])

        return arr

    def pivot1(self, tableau: 'BranchAndBound', row: int, col: int) -> 'BranchAndBound':
        tableau._Simplexe__tableau[row, :] /= tableau._Simplexe__tableau[row, col]
        tableau._Simplexe__tableau[np.arange(tableau._Simplexe__tableau.shape[0]) != row, :] -= tableau._Simplexe__tableau[np.arange(tableau._Simplexe__tableau.shape[0]) != row, col][:, np.newaxis] * tableau._Simplexe__tableau[row, :]


    def pivot(self, value: np.ndarray, row: int, col: int) -> None:
        pivot_row: np.ndarray = value[row, :]
        pivot_row /= pivot_row[col]
        rows_to_update = np.arange(value.shape[0]) != row
        value[rows_to_update, :] -= value[rows_to_update, col, np.newaxis] * pivot_row

    def solve_tableau(self, tableau: 'BranchAndBound') -> 'BranchAndBound':
        while True:
            row, col = self.find_pivot(tableau)
            if row is None:
                break
            t1 = time.time()
            self.pivot(tableau._Simplexe__tableau, row, col)
            t2 = time.time()
            print("pivot time",t2-t1)
        tableau = self.round_numpy_array(tableau)
        return tableau

    def find_pivot(self, tableau: 'BranchAndBound') -> tuple[int, int]:
        # Find the first row with a negative right-hand-side value
        rhs = tableau._Simplexe__tableau[:-1, -1]
        rows_with_negative_rhs = rhs < 0
        if not np.any(rows_with_negative_rhs):
            return None, None
        row_with_min_rhs = np.argmin(rhs[rows_with_negative_rhs])
        pivot_row = np.arange(tableau._Simplexe__tableau.shape[0] - 1)[rows_with_negative_rhs][row_with_min_rhs]

        # Find the column with the highest ratio of column value to objective function value
        cols_with_negative_entries = tableau._Simplexe__tableau[pivot_row, :-1] < 0
        if not np.any(cols_with_negative_entries):
            return None, None
        ratios = tableau._Simplexe__tableau[-1, :-1][cols_with_negative_entries] / tableau._Simplexe__tableau[pivot_row, :-1][cols_with_negative_entries]
        pivot_col = np.arange(tableau._Simplexe__tableau.shape[1] - 1)[cols_with_negative_entries][np.argmax(ratios)]

        return pivot_row, pivot_col


if "__main__" == __name__:
  bb = BranchAndBound()
  bb.go("lp_sample.txt")