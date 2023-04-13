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
        self.LoadFromFile(lpFileName, printDetails) # loads file and solves it with the simplex method
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
        def is_almost_integer(num, threshold=0.0001):
            return abs(num - round(num)) <= threshold
        def update_nodes(node, list_node):
            x_column = list(range(len(node._Simplexe__tableau[0]) - len(node._Simplexe__tableau)))

            for col in x_column:
                for row in range(len(node._Simplexe__tableau) - 1):
                    if node._Simplexe__tableau[row][col] == 1 and (abs(node._Simplexe__tableau[row][-1]) - abs(floor(node._Simplexe__tableau[row][-1]))) > 1e-6:
                        node.index.append(row)

            for line in node.index:
                left_tableau = self.create_bounds(deepcopy(node), line, True)
                right_tableau = self.create_bounds(deepcopy(node), line, False)
                list_node.append(left_tableau)
                list_node.append(right_tableau)

            node.index = []
            return list_node

        numRows, numCols = self._Simplexe__tableau.shape
        temp_node = deepcopy(self)

        list_node = []
        list_sol = []
        z_PLNE = float('-inf')

        list_node = update_nodes(temp_node, list_node)


        while list_node:
            node = list_node.pop(0)

            node._Simplexe__iteration = 0
            node.OptStatus = OptStatus.Unknown
            node.PrintTableau("after bounds addition, before pivoting")
            #node._Simplexe__tableau = self.solve_tableau(node._Simplexe__tableau)
            #node.PrintTableau("after 0 pivots")
            node._Simplexe__tableau = self.solve_tableau(node._Simplexe__tableau)
            node.PrintTableau("after 1 pivots")


            objval = node.ObjValue()
            print("objVal after pivots", objval)

            isInteger = is_almost_integer(objval) and not any(t < 0 for t in node._Simplexe__tableau[:-1, -1])
            if isInteger:
                z_PLNE = max(z_PLNE, objval)
                list_sol.append(node._Simplexe__tableau)
                print("solution found")
                node.PrintTableau("Branch and Bound Solution")
                print("OBJECTIVE VALUE : {:.2f}".format(z_PLNE))
                sys.exit(0)
            else:
                list_node = update_nodes(node, list_node)



    def pivot(self, tableau, row, col):
        tableau[row, :] /= tableau[row, col]
        for i in range(tableau.shape[0]):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]

    def solve_tableau(self, tableau):
        while True:
            row, col = self.find_pivot(tableau)
            print(row, col)
            if row is None:
                break
            self.pivot(tableau, row, col)
        return tableau

    def find_pivot(self, tableau):
        row_with_min = np.argmin(tableau[:-1, 2])
        col = np.argmin(tableau[row_with_min, :-1])
        if tableau[row_with_min, col] >= 0:
            return None, None

        rows_with_positive_coeff = tableau[:-1, col] > 0
        if not np.any(rows_with_positive_coeff):
            return None, None

        ratio = tableau[:-1, -1][rows_with_positive_coeff] / tableau[:-1, col][rows_with_positive_coeff]
        row = np.argmin(ratio)
        return np.arange(tableau.shape[0] - 1)[rows_with_positive_coeff][row], col


if "__main__" == __name__:
  bb = BranchAndBound()
  bb.go("lp_sample.txt")