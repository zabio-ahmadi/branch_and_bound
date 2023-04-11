import numpy as np
from os.path import isfile
import time, sys, math
from enum import Enum
from copy import deepcopy

from simplexe import *
from constants import *
from math import floor, ceil


class BranchAndBound(Simplexe):
    def __init__(self):
        super().__init__()
        self.index = []

    def go(self, lpFileName, printDetails=False):
        self.LoadFromFile(lpFileName, printDetails)
        self.PLNE()

    def solvePhase1_v2(self):
        phaseISplx = Simplexe()
        self.initPhaseI_v2(phaseISplx)
        if phaseISplx.ObjValue() > Constants.EPS:
            self.OptStatus = OptStatus.Infeasible
            return
        if self.__PrintDetails:
            phaseISplx.PrintTab("Optimal solution Phase I")
        phaseISplx.RemoveAuxiliaryBasicVariables()
        phaseISplx.PrintSolution()
        self._Simplexe__tab = phaseISplx._Simplexe__tab[0:self.NumRows, 0:self.NumCols+self.NumRows]
        self._Simplexe__tab = np.append(self._Simplexe__tab, np.array([phaseISplx._Simplexe__tab[-2, 0:self.NumCols+self.NumRows]]), axis=0)
        self._Simplexe__tab = np.append(self._Simplexe__tab, np.array([phaseISplx._Simplexe__tab[0:-1, -1]]).T, axis=1)
        self._Simplexe__basicVars = phaseISplx._Simplexe__basicVars[0:-1]
        self._Simplexe__PhaseIPivotCount = phaseISplx._Simplexe__PivotCount
        self.OptStatus = OptStatus.Feasible

    def initPhaseI_v2(self, phaseISplx):
        phaseISplx.IsPhaseI, phaseISplx._Simplexe__start = True, time.time()
        phaseISplx.NumRows, phaseISplx.NumCols = self.NumRows + 1, self.NumCols + self.NumRows
        phaseISplx.OptStatus = OptStatus.Feasible
        tmpCopy = self._Simplexe__tab.copy()
        phaseISplx._Simplexe__tab = tmpCopy[0:-1,0:-1]
        phaseISplx._Simplexe__tab = np.append(phaseISplx._Simplexe__tab, np.identity(self.NumRows), axis=1)
        phaseISplx._Simplexe__tab = np.append(phaseISplx._Simplexe__tab, np.array([tmpCopy[0:-1, -1]]).T, axis=1)
        objRowOriginal, objRowNew = np.zeros(phaseISplx.NumRows + phaseISplx.NumCols, dtype=float), np.zeros(phaseISplx.NumRows + phaseISplx.NumCols, dtype=float)
        for iCol in range(phaseISplx.NumCols):
            objRowOriginal[iCol] = tmpCopy[-1, iCol]
            objRowNew[iCol] = -np.sum(phaseISplx._Simplexe__tab[:, iCol])
        objRowNew[-1] = -np.sum(phaseISplx._Simplexe__tab[:, -1])
        phaseISplx._Simplexe__tab = np.append(phaseISplx._Simplexe__tab, np.array([objRowOriginal]), axis=0)
        phaseISplx._Simplexe__tab = np.append(phaseISplx._Simplexe__tab, np.array([objRowNew]), axis=0)
        phaseISplx._Simplexe__basicVars = np.arange(phaseISplx.NumCols, phaseISplx.NumCols + phaseISplx.NumRows, 1, dtype=int)
        phaseISplx._Simplexe__basicVars[-1] = -1
        phaseISplx.TabRowCount, phaseISplx.TabColCount = phaseISplx._Simplexe__tab.shape
        if phaseISplx._Simplexe__PrintDetails: phaseISplx.PrintTab("Phase 1 Tab")
        phaseISplx._Simplexe__performPivots()

    def create_bounds(self, tableau, index, isLeft=True):
        if isLeft:
            abs_val = floor(tableau[index][-1])
            new_line = [1] + [0] * (len(tableau[0]) - 2) + [1] + [abs_val]
        else:
            abs_val = -1 * ceil(tableau[index][-1])
            new_line = [-1] + [0] * (len(tableau[0]) - 2) + [1] + [abs_val]

        tableau = np.hstack((tableau[:, :-1], np.atleast_2d([0] * len(tableau)).T, tableau[:, -1:]))
        tableau = np.vstack((tableau[:-1], new_line, tableau[-1:]))

        for i in range(len(tableau[index])):
            if isLeft:
                tableau[-2][i] = tableau[-2][i] - tableau[index][i]
            else:
                tableau[-2][i] = tableau[-2][i] + tableau[index][i]
        return tableau

    def PLNE(self):
        numCols = len(self._Simplexe__tab[0])
        numRows = len(self._Simplexe__tab)

        temp = np.copy(self._Simplexe__tab)
        last_column = temp[:-1, -1:]

        x_column = list(range(numCols - numRows))

        for col in x_column:
            for row in range(len(temp) - 1):
                if temp[row][col] == 1 and (abs(temp[row][-1]) - abs(floor(temp[row][-1]))) > 1e-6:
                    self.index.append(row)

        list_node = []

        for line in self.index:
            left_tableau = self.create_bounds(temp, line, True)
            right_tableau = self.create_bounds(temp, line, False)

            list_node.append(left_tableau)
            list_node.append(right_tableau)

        self.index = []

        z_PLNE = float('inf')
        list_sol = []

        while list_node:
            self._Simplexe__tab = node = list_node.pop(0)

            self._Simplexe__iteration = 0
            self.solvePhase1_v2()
            self._Simplexe__tab = self._Simplexe__solveTableau(self._Simplexe__tab)

            last_column = self._Simplexe__tab[:-1, -1:]

            isInteger = True
            for x in last_column:
                if abs(x[0]) == np.inf and not (x[0] - round(x[0])) < 1e-6:
                    isInteger = False

            if isInteger and not any(t < 0 for t in self._Simplexe__tab[:-1, -1]):
                z_PLNE = min(z_PLNE, self.ObjValue())
                list_sol.append(self._Simplexe__tab)
                print("solution found")
                self.PrintTab("Branch and Bound Solution")
                print("OBJECTIVE VALUE : {:.2f}".format(self.ObjValue()))
                sys.exit(0)
            else:
                x_column = list(range(len(self._Simplexe__tab[0]) - len(self._Simplexe__tab)))

                for col in x_column:
                 for row in range(len(self._Simplexe__tab) - 1):
                  if self._Simplexe__tab[row][col] == 1 and (abs(self._Simplexe__tab[row][-1]) - abs(floor(self._Simplexe__tab[row][-1]))) > 1e-6:
                   self.index.append(row)
            for line in self.index:
                left_tableau = self.create_bounds(self._Simplexe__tab, line, True)
                right_tableau = self.create_bounds(self._Simplexe__tab, line, False)
                list_node.append(left_tableau)
                list_node.append(right_tableau)
            self.index = []



if "__main__" == __name__:
  bb = BranchAndBound()
  bb.go("lp_sample.txt")