import numpy as np
from os.path import isfile
import time, sys, math
from enum import Enum
from copy import deepcopy

from src.simplexe import *
from src.constants import *

from math import floor, ceil
from typing import *

class BranchAndBound(Simplexe):
    def __init__(self):
        super().__init__()
        self.index = []
        self.depth = 0

    def go(self, lpFileName, printDetails=False):
        self._Simplexe__PrintDetails = printDetails
        self.LoadFromFile(lpFileName, printDetails) # loads file and solves it with the simplex method
        #self = self.round_numpy_array(self)
        self.PrintTableau("before PLNE")
        self.PLNE()
        print("------------- finish PLNE")
        #self.PrintSolution()

    def create_bounds(self, tableau: 'BranchAndBound', isLeft=True):
        whichRow = tableau.index[tableau.depth]
        whichVariable = tableau.depth
        rhs_val = tableau._Simplexe__tableau[whichRow][-1]
        abs_val = floor(rhs_val) if isLeft else -1 * ceil(rhs_val)
        sign = 1 if isLeft else -1
        tableau.NumRows += 1
        AMatrix_row = np.zeros(tableau.LP.AMatrix.shape[1])
        AMatrix_row[whichVariable] = sign
        tableau.LP.AMatrix = np.vstack((tableau.LP.AMatrix, AMatrix_row))
        tableau.LP.RHS = np.append(tableau.LP.RHS, abs_val)

        return tableau

    def PLNE(self):
        def is_almost_integer(num, threshold=0.01):
            frac_part = abs(num - round(num))
            return frac_part <= threshold or (1 - frac_part) <= threshold


        def update_nodes(node, list_node, depth=0):
            node.index = [i for i in range(len(node._Simplexe__basicVariables)) if node._Simplexe__basicVariables[i] < node.NumCols]

            left_tableau = self.create_bounds(deepcopy(node), True)
            left_tableau.depth += 1
            right_tableau = self.create_bounds(deepcopy(node), False)
            right_tableau.depth += 1
            list_node.append(left_tableau)
            list_node.append(right_tableau)

            node.index = []
            return list_node

        numRows, numCols = self._Simplexe__tableau.shape

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
            #node.PrintTableau("before solve")
            node.OptStatus == OptStatus.Unknown
            #node.initPhaseI(deepcopy(node))
            #node._Simplexe__solvePhaseI()
            node._Simplexe__solveProblem(OptStatus.Unknown)
            #node = self.solve_tableau(node)
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

    def round_numpy_array(self, arr: 'BranchAndBound', decimals=6):
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


    def pivot(self, tableau: 'BranchAndBound', row, col):
        tableau._Simplexe__tableau[row, :] /= tableau._Simplexe__tableau[row, col]
        for i in range(tableau._Simplexe__tableau.shape[0]):
            if i != row:
                tableau._Simplexe__tableau[i, :] -= tableau._Simplexe__tableau[i, col] * tableau._Simplexe__tableau[row, :]

    def solve_tableau(self, tableau: 'BranchAndBound'):
        while True:
            row, col = self.find_pivot(tableau)
            if row is None:
                break
            self.pivot(tableau, row, col)
        #tableau = self.round_numpy_array(tableau)
        return tableau

    def find_pivot(self, tableau: 'BranchAndBound'):
        #tableau.PrintTableau("find pivot")
        mask = tableau._Simplexe__basicVariables >= tableau.NumCols
        t = tableau._Simplexe__tableau[:-1]
        row_with_min = np.argmin(t[mask, tableau.NumCols-1])
        col = np.argmin(tableau._Simplexe__tableau[row_with_min, :-1])
        if tableau._Simplexe__tableau[row_with_min, col] >= 0:
            return None, None

        rows_with_positive_coeff = tableau._Simplexe__tableau[:-1, col] > 0
        if not np.any(rows_with_positive_coeff):
            return None, None

        ratio = tableau._Simplexe__tableau[:-1, -1][rows_with_positive_coeff] / tableau._Simplexe__tableau[:-1, col][rows_with_positive_coeff]
        row = np.argmin(ratio)
        return np.arange(tableau._Simplexe__tableau.shape[0] - 1)[rows_with_positive_coeff][row], col


if "__main__" == __name__:
  bb = BranchAndBound()
  bb.go("lp_sample.txt")