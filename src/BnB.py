import numpy as np
from os.path import isfile
import time, sys, math
from enum import Enum
from copy import deepcopy

from src.simplexe import *
from src.constants import *

from typing import *

class BranchAndBound(Simplexe):
    def __init__(self):
        super().__init__()
        self.index: List = []
        self.depth: int = 0

    def go(self, lpFileName: str, printDetails: bool = False) -> None:
        self._Simplexe__PrintDetails = printDetails
        self.LoadFromFile(lpFileName, printDetails) # loads file and solves it with the simplex method
        #self = self.round_numpy_array(self)
        self.PrintTableau("before PLNE")
        self.PLNE()
        print("------------- finish PLNE")
        #self.PrintSolution()

    def create_bounds(self, node: 'BranchAndBound', isLeft: bool = True) -> None:
        tab: np.ndarray = node._Simplexe__tableau
        whichRow = node.index[node.depth % len(node.index)]
        whichVariable = node.depth
        rhs_val = tab[whichRow][-1]
        abs_val = np.floor(rhs_val) if isLeft else -1 * np.ceil(rhs_val)
        new_line = [0] * (tab.shape[1] - 1) + [1] + [abs_val]
        sign = 1.0 if isLeft else -1.0
        new_line[whichVariable] = sign

        tab = np.hstack((tab[:, :-1], np.atleast_2d([0] * tab.shape[0]).T, tab[:, -1:]))

        tab = np.vstack((tab[:-1], new_line, tab[-1:]))

        if isLeft:
            tab[-2] -= tab[whichRow]
        else:
            tab[-2] += tab[whichRow]
        node.NumCols += 1
        node.NumRows += 1
        node._Simplexe__basicVariables = np.append(node._Simplexe__basicVariables, np.max(node._Simplexe__basicVariables)+1)
        node._Simplexe__basicVariables = np.where(node._Simplexe__basicVariables >= node.NumCols, node._Simplexe__basicVariables + 1, node._Simplexe__basicVariables)
        node._Simplexe__tableau = tab


    def PLNE(self):
        def is_almost_integer(num: float, threshold: float = 0.01) -> bool:
            return abs(num - round(num)) < threshold

        def update_nodes(node: 'BranchAndBound', list_node: List) -> List:
            node.index = [i for i in range(len(node._Simplexe__basicVariables)) if node._Simplexe__basicVariables[i] < node.NumCols]
            left_tableau, right_tableau = deepcopy(node), deepcopy(node)
            self.create_bounds(left_tableau, True)
            self.create_bounds(right_tableau, False)
            left_tableau.depth += 1
            right_tableau.depth += 1
            list_node.append(left_tableau)
            list_node.append(right_tableau)

            node.index = []
            return list_node

        objval_max = self.ObjValue()
        temp_node = deepcopy(self)
        list_node, list_sol, z_PLNE = [], [], float('-inf')
        iteration, max_iterations = 0, 1000
        list_node = update_nodes(temp_node, list_node)
        best_tableau = None


        while list_node and iteration < max_iterations:
            iteration += 1
            node = list_node.pop(0)
            node = self.solve_tableau(node)
            node.PrintTableau("after solve")

            objval = node.ObjValue()
            print("objVal after pivots", objval)

            if objval < z_PLNE or objval > objval_max: # no need to check anything if current objective value is worse than best
                continue

            isInteger = is_almost_integer(objval) and not any(t < -1e-6 for t in node._Simplexe__tableau[:-1, -1])
            if isInteger:
                if objval > z_PLNE:
                    z_PLNE = objval
                    best_tableau = node
                    list_sol.append(node)
                    print("Better solution found")
                    #node.PrintTableau("Branch and Bound Solution")
                    print("OBJECTIVE VALUE : {:.2f}".format(z_PLNE))
                #sys.exit(0)
            elif all(t >= 0 for t in node._Simplexe__tableau[:-1, -1]):  # Update nodes only when the current node has a feasible solution
                list_node = update_nodes(node, list_node)

        # Print the best solution found after searching all nodes
        if best_tableau:
            print("\nBest solution found")
            best_tableau.PrintTableau("Best BnB Solution")
            print("OBJECTIVE VALUE : {:.2f}".format(z_PLNE))


    def round_numpy_array(self, arr: np.ndarray, decimals: int = 6) -> np.ndarray:
        # convert the array to a floating-point type
        arr = arr.astype(float)

        # round the array to a maximum of 6 decimal places
        arr = np.round(arr, decimals=decimals)

        # convert any non-float values back to strings
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if not isinstance(arr[i, j], float):
                    arr[i, j] = str(arr[i, j])

        return arr

    def pivot(self, tab: np.ndarray, row: int, col: int) -> None:
        pivot_row: np.ndarray = tab[row, :]
        pivot_row /= pivot_row[col]
        rows_to_update = np.arange(tab.shape[0]) != row
        tab[rows_to_update, :] -= tab[rows_to_update, col, np.newaxis] * pivot_row

    def solve_tableau(self, tableau: 'BranchAndBound') -> 'BranchAndBound':
        tab: np.ndarray = tableau._Simplexe__tableau
        while True:
            row, col = self.find_pivot(tab)
            if row is None:
                break
            t1 = time.time()
            self.pivot(tab, row, col)
            t2 = time.time()
            print("pivot time",t2-t1)
        tab = self.round_numpy_array(tab)
        return tableau

    def find_pivot(self, tableau: np.ndarray) -> tuple[int, int]:
        # Find the first row with a negative right-hand-side value
        rhs = tableau[:-1, -1]
        rows_with_negative_rhs = rhs < 0
        if not np.any(rows_with_negative_rhs):
            return None, None
        row_with_min_rhs = np.argmin(rhs[rows_with_negative_rhs])
        pivot_row = np.arange(tableau.shape[0] - 1)[rows_with_negative_rhs][row_with_min_rhs]

        # Find the column with the highest ratio of column value to objective function value
        cols_with_negative_entries = tableau[pivot_row, :-1] < 0
        if not np.any(cols_with_negative_entries):
            return None, None
        ratios = tableau[-1, :-1][cols_with_negative_entries] / tableau[pivot_row, :-1][cols_with_negative_entries]
        pivot_col = np.arange(tableau.shape[1] - 1)[cols_with_negative_entries][np.argmax(ratios)]

        return pivot_row, pivot_col


if "__main__" == __name__:
  bb = BranchAndBound()
  bb.go("lp_sample.txt")