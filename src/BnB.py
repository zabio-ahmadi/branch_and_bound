import numpy as np
from os.path import isfile
import time, sys, math
from enum import Enum
from copy import deepcopy

from src.simplexe import *
from src.constants import *

from math import floor, ceil


class BranchAndBound(Simplexe):
    def __init__(self):
        super().__init__()
        self.index = []

    def go(self, lpFileName, printDetails=False):
        self.__PrintDetails = printDetails
        self.LoadFromFile(lpFileName, printDetails)
        self.PLNE()
        print("------------- finish PLNE")
        self.PrintSolution()

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
        self._Simplexe__tableau = phaseISplx._Simplexe__tableau[0:self.NumRows, 0:self.NumCols+self.NumRows]
        self._Simplexe__tableau = np.append(self._Simplexe__tableau, np.array([phaseISplx._Simplexe__tableau[-2, 0:self.NumCols+self.NumRows]]), axis=0)
        tmp1 = np.array([phaseISplx._Simplexe__tableau[0:-1, -1]]).T
        self._Simplexe__tableau = np.append(self._Simplexe__tableau, np.array([phaseISplx._Simplexe__tableau[0:-1, -1]]).T, axis=1)
        self._Simplexe__basicVariables = phaseISplx._Simplexe__basicVariables[0:-1]
        self._Simplexe__PhaseIPivotCount = phaseISplx._Simplexe__PivotCount
        self.OptStatus = OptStatus.Feasible

    def initPhaseI_v2(self, phaseISplx):
        phaseISplx.IsPhaseI, phaseISplx._Simplexe__start = True, time.time()
        phaseISplx.NumRows, phaseISplx.NumCols = self.NumRows + 1, self.NumCols + self.NumRows + 2
        phaseISplx.OptStatus = OptStatus.Feasible
        tmpCopy = self._Simplexe__tableau.copy()
        phaseISplx._Simplexe__tableau = tmpCopy[0:-1,0:-1]
        idnty = np.identity(phaseISplx.NumRows)
        phaseISplx._Simplexe__tableau = np.append(phaseISplx._Simplexe__tableau, idnty, axis=1)
        phaseISplx._Simplexe__tableau = np.append(phaseISplx._Simplexe__tableau, np.array([tmpCopy[0:-1, -1]]).T, axis=1)
        objRowOriginal, objRowNew = np.zeros(phaseISplx.NumRows + phaseISplx.NumCols, dtype=float), np.zeros(phaseISplx.NumRows + phaseISplx.NumCols, dtype=float)
        for iCol in range(phaseISplx.NumCols):
            objRowOriginal[iCol] = tmpCopy[-1, iCol]
            objRowNew[iCol] = -np.sum(phaseISplx._Simplexe__tableau[:, iCol])
        objRowNew[-1] = -np.sum(phaseISplx._Simplexe__tableau[:, -1])
        phaseISplx._Simplexe__tableau = np.append(phaseISplx._Simplexe__tableau, np.array([objRowOriginal]), axis=0)
        phaseISplx._Simplexe__tableau = np.append(phaseISplx._Simplexe__tableau, np.array([objRowNew]), axis=0)
        phaseISplx._Simplexe__basicVariables = np.arange(phaseISplx.NumCols, phaseISplx.NumCols + phaseISplx.NumRows, 1, dtype=int)
        phaseISplx._Simplexe__basicVariables[-1] = -1
        phaseISplx.TableauRowCount, phaseISplx.TableauColCount = phaseISplx._Simplexe__tableau.shape
        if phaseISplx._Simplexe__PrintDetails: phaseISplx.PrintTab("Phase 1 Tab")
        phaseISplx._Simplexe__performPivots()

    def create_bounds1(self, tableau, index, isLeft=True):
        if isLeft:
            abs_val = floor(tableau[index][-1])
            new_line = [1] + [0] * (len(tableau[0]) - 2) + [1] + [abs_val]
        else:
            abs_val = -1 * ceil(tableau[index][-1])
            new_line = [-1] + [0] * (len(tableau[0]) - 2) + [1] + [abs_val]
        return new_line
    def create_bounds(self, tableau, index, isLeft=True):

        if isLeft == True:
            abs = floor(tableau[index][-1])
            new_line = [1] + [0] * (len(tableau[0]) - 2) + [1] + [abs]

        if isLeft == False:
            abs = -1 * ceil(tableau[index][-1])
            new_line = [-1] + [0] * (len(tableau[0]) - 2) + [1] + [abs]

        #tableau = np.delete(tableau, index, axis=0)
        tableau = np.hstack((tableau[:, :-1], np.atleast_2d([0] * len(tableau)).T, tableau[:, -1:]))

        tableau = np.vstack((tableau[:-1], new_line, tableau[-1:]))

        for i in range(len(tableau[index])):
            if isLeft == True:
                tableau[-2][i] = tableau[-2][i] - tableau[index][i]
            if isLeft == False:
                tableau[-2][i] = tableau[-2][i] + tableau[index][i]
        return tableau

    def PLNE(self):
        numCols = len(self._Simplexe__tableau[0])
        numRows = len(self._Simplexe__tableau)

        temp = np.copy(self._Simplexe__tableau)
        last_column = temp[:-1, -1:]

        x_column = list(range(numCols - numRows))

        for col in x_column:
            for row in range(len(temp) - 1):
                if temp[row][col] == 1 and (abs(temp[row][-1]) - abs(floor(temp[row][-1]))) > 1e-6:
                    self.index.append(row)

        #list_node = [(temp, [])]
        list_node = []

        for line in self.index:
            left_tableau = self.create_bounds(np.copy(temp), line, True)
            right_tableau = self.create_bounds(np.copy(temp), line, False)
            NumRowsCols = (self.NumRows + 1, self.NumCols + 1)

            list_node.append((left_tableau, NumRowsCols))
            list_node.append((right_tableau, NumRowsCols))

        self.index = []

        z_PLNE = float('inf')
        list_sol = []

        while list_node:
            self._Simplexe__tableau, NumRowsCols = list_node.pop(0)
            self.NumRows, self.NumCols = NumRowsCols[0], NumRowsCols[1]
            #temp, added_constraints = node = list_node.pop(0)
            #self._Simplexe__tableau = np.copy(temp)

            #for constraint in added_constraints:
            #    self._Simplexe__tableau = np.hstack((self._Simplexe__tableau[:, :-1], np.atleast_2d([0] * len(self._Simplexe__tableau)).T, self._Simplexe__tableau[:, -1:]))
            #    self._Simplexe__tableau = np.vstack((self._Simplexe__tableau[:-1], constraint, self._Simplexe__tableau[-1:]))

            self._Simplexe__iteration = 0
            self._Simplexe__solveProblem()
            #self._Simplexe__solvePhaseI()
            #self._Simplexe__tableau = self._Simplexe__solveProblem(self._Simplexe__tableau)

            last_column = self._Simplexe__tableau[:-1, -1:]
            objval = self.ObjValue()
            if objval < 1e-6:
                isInteger = True
            else:
                isInteger = False

            #isInteger = True
            #for x in last_column:
            #    if abs(x[0]) == np.inf and not (x[0] - round(x[0])) < 1e-6:
            #        isInteger = False

            if isInteger and not any(t < 0 for t in self._Simplexe__tableau[:-1, -1]):
                z_PLNE = min(z_PLNE, self.ObjValue())
                list_sol.append(self._Simplexe__tableau)
                print("solution found")
                self.PrintTableau("Branch and Bound Solution")
                print("OBJECTIVE VALUE : {:.2f}".format(self.ObjValue()))
                sys.exit(0)
            else:
                x_column = list(range(len(self._Simplexe__tableau[0]) - len(self._Simplexe__tableau)))

                for col in x_column:
                 for row in range(len(self._Simplexe__tableau) - 1):
                  if self._Simplexe__tableau[row][col] == 1 and (abs(self._Simplexe__tableau[row][-1]) - abs(floor(self._Simplexe__tableau[row][-1]))) > 1e-6:
                   self.index.append(row)
            for line in self.index:
                left_tableau = self.create_bounds(np.copy(self._Simplexe__tableau), line, True)
                right_tableau = self.create_bounds(np.copy(self._Simplexe__tableau), line, False)
                NumRowsCols = (self.NumRows + 1, self.NumCols + 1)
                list_node.append((left_tableau, NumRowsCols))
                list_node.append((right_tableau, NumRowsCols))
                #left_constraint = self.create_bounds(self._Simplexe__tableau, line, True)
                #right_constraint = self.create_bounds(self._Simplexe__tableau, line, False)
                #list_node.append((temp, added_constraints + [left_constraint]))
                #list_node.append((temp, added_constraints + [right_constraint]))
            self.index = []



if "__main__" == __name__:
  bb = BranchAndBound()
  bb.go("lp_sample.txt")