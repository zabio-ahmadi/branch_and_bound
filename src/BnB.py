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

    def go(self, lpFileName, printDetails=False):
        self._Simplexe__PrintDetails = printDetails
        self.LoadFromFile(lpFileName, printDetails)
        self.PrintTableau("before PLNE")
        self.PLNE()
        print("------------- finish PLNE")
        self.PrintSolution()
    
    
    def create_bounds(self, tableau: Type['BranchAndBound'], index, isLeft=True):

        if isLeft == True:
            abs = floor(tableau._Simplexe__tableau[index][-1])
            new_line = [1] + [0] * (len(tableau._Simplexe__tableau[0]) - 2) + [1] + [abs]

        if isLeft == False:
            abs = -1 * ceil(tableau._Simplexe__tableau[index][-1])
            new_line = [-1] + [0] * (len(tableau._Simplexe__tableau[0]) - 2) + [1] + [abs]

        #tableau = np.delete(tableau, index, axis=0)
        tableau._Simplexe__tableau = np.hstack((tableau._Simplexe__tableau[:, :-1], np.atleast_2d([0] * len(tableau._Simplexe__tableau)).T, tableau._Simplexe__tableau[:, -1:]))

        tableau._Simplexe__tableau = np.vstack((tableau._Simplexe__tableau[:-1], new_line, tableau._Simplexe__tableau[-1:]))

        for i in range(len(tableau._Simplexe__tableau[index])):
            if isLeft == True:
                tableau._Simplexe__tableau[-2][i] = tableau._Simplexe__tableau[-2][i] - tableau._Simplexe__tableau[index][i]
            if isLeft == False:
                tableau._Simplexe__tableau[-2][i] = tableau._Simplexe__tableau[-2][i] + tableau._Simplexe__tableau[index][i]
        tableau.NumCols += 1
        tableau.NumRows += 1
        tableau._Simplexe__basicVariables = np.arange(tableau.NumCols, tableau.NumCols + tableau.NumRows, 1, dtype=int)
        return tableau

    def PLNE(self):
        numRows, numCols = self._Simplexe__tableau.shape

        temp_node = deepcopy(self)
        last_column = temp_node._Simplexe__tableau[:-1, -1:]
        

        x_column = list(range(numCols - numRows))

        for col in x_column:
            for row in range(temp_node._Simplexe__tableau.shape[0] - 1):
                if temp_node._Simplexe__tableau[row][col] == 1 and (abs(temp_node._Simplexe__tableau[row][-1]) - abs(floor(temp_node._Simplexe__tableau[row][-1]))) > 1e-6:
                    self.index.append(row)

        #list_node = [(temp, [])]
        list_node = []

        for line in self.index:
            left_tableau = self.create_bounds(deepcopy(temp_node), line, True)
            right_tableau = self.create_bounds(deepcopy(temp_node), line, False)

            list_node.append(left_tableau)
            list_node.append(right_tableau)

        self.index = []

        z_PLNE = float('inf')
        list_sol = []

        while list_node:
            node = list_node.pop(0)

            node._Simplexe__iteration = 0
            node.OptStatus = OptStatus.Unknown
            node.PrintTableau("after bounds addition, before pivoting")
            #node._Simplexe__solveProblem()
            #node._Simplexe__solvePhaseI()
            #node._Simplexe__performPivots()
            node._Simplexe__tableau = self.solve_tableau(node._Simplexe__tableau, 0)
            node._Simplexe__tableau = self.solve_tableau(node._Simplexe__tableau, 2)
            node.PrintTableau("after pivots")

            last_column = node._Simplexe__tableau[:-1, -1:]
            objval = node.ObjValue()
            if objval < 1e-6:
                isInteger = True
            else:
                isInteger = False

            #isInteger = True
            #for x in last_column:
            #    if abs(x[0]) == np.inf and not (x[0] - round(x[0])) < 1e-6:
            #        isInteger = False

            if isInteger and not any(t < 0 for t in node._Simplexe__tableau[:-1, -1]):
                z_PLNE = min(z_PLNE, node.ObjValue())
                list_sol.append(node._Simplexe__tableau)
                print("solution found")
                self.PrintTableau("Branch and Bound Solution")
                print("OBJECTIVE VALUE : {:.2f}".format(self.ObjValue()))
                sys.exit(0)
            else:
                x_column = list(range(len(node._Simplexe__tableau[0]) - len(node._Simplexe__tableau)))

                for col in x_column:
                 for row in range(len(node._Simplexe__tableau) - 1):
                  if node._Simplexe__tableau[row][col] == 1 and (abs(node._Simplexe__tableau[row][-1]) - abs(floor(node._Simplexe__tableau[row][-1]))) > 1e-6:
                   node.index.append(row)
            for line in self.index:
                left_tableau = self.create_bounds(deepcopy(node), line, True)
                right_tableau = self.create_bounds(deepcopy(node), line, False)
                list_node.append(left_tableau)
                list_node.append(right_tableau)
                #left_constraint = self.create_bounds(self._Simplexe__tableau, line, True)
                #right_constraint = self.create_bounds(self._Simplexe__tableau, line, False)
                #list_node.append((temp, added_constraints + [left_constraint]))
                #list_node.append((temp, added_constraints + [right_constraint]))
            self.index = []

    def find_pivot(self, tableau, two: bool):
        if two:
            col = np.argmin(tableau[-1, :-1])
            if tableau[-1, col] >= 0:
                return None, None
        else:
            col = np.argmin(tableau[1, :-1])
            if tableau[1, col] >= 0:
                return None, None

        rows_with_positive_coeff = tableau[:-1, col] > 0
        if not np.any(rows_with_positive_coeff):
            return None, None

        ratio = tableau[:-1, -1][rows_with_positive_coeff] / tableau[:-1, col][rows_with_positive_coeff]
        row = np.argmin(ratio)
        return np.arange(tableau.shape[0] - 1)[rows_with_positive_coeff][row], col

    def pivot(self, tableau, row, col):
        tableau[row, :] /= tableau[row, col]
        for i in range(tableau.shape[0]):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]

    def solve_tableau(self, tableau, two: bool): # two is for alternating between choosing row 1 and last row for pivot selection
        while True:
            row, col = self.find_pivot(tableau, two)
            if row is None:
                break
            self.pivot(tableau, row, col)

        return tableau



if "__main__" == __name__:
  bb = BranchAndBound()
  bb.go("lp_sample.txt")