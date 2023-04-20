"""
  Author:    Niklaus Eggenberg
  Created:   23.02.2023
  Simplex Class - contains the code to solve an LP via simplex method
"""
#run command: 
# cd 
#python c:\HEPIA\neg-teaching-material\Simplexe\main.py C:\HEPIA\neg-teaching-material\2022-2023\2e\lp_test.txt
import numpy as np
import os.path
from os import path
from src.constants import Constants, OptimizationType, OptStatus, PivotMode
from src.lp import LP
import time
import sys

class Simplexe:
  def __init__(self):
    self.IsPhaseI = False
    self.__isRemovingAuxVariables = False
    self.__PivotCount = 0
    self.__PhaseIPivotCount = 0
    self.__PrintDetails = False

  def LoadFromFile(self, lpFileName, printDetails):
    print("-----entering LoadFromFile")
    self.FileName = lpFileName
    self.__PrintDetails = printDetails == True
    self.LP = LP(lpFileName)
    if not self.LP.ParseFile():
        return
    # --- 
    self.OptStatus = OptStatus.Unknown
    self.NumRows = self.LP.RHS.shape[0]
    self.NumCols = self.LP.Costs.shape[0]
    self.LP.PrintProblem()
    self.__solveProblem()

  def LoadFromString(self, inputString, printDetails):
    print("-----entering LoadFromString")
    self.__PrintDetails = printDetails
    self.LP = LP(inputString, True)
    if not self.LP.ParseString():
        return
    # --- 
    self.OptStatus = OptStatus.Unknown
    self.NumRows = self.LP.RHS.shape[0]
    self.NumCols = self.LP.Costs.shape[0]
    self.LP.PrintProblem()
    self.__solveProblem()


  def __solveProblem(self, optstatus=-1):
    if optstatus != -1:
      self.OptStatus = OptStatus.Unknown
    print("-----entering __solveProblem", self.OptStatus)
    self.__start = time.time()
    if self.OptStatus == OptStatus.Unknown: self.__initTableau()
    # make sure we have a feasible solution - go to PhaseI
    # phase I runs only if 
    if (self.__isSolutionFeasible() != True):
        print("Initial solution is unfeasible - starting Phase I...")
        self.__solvePhaseI()
        if self.OptStatus == OptStatus.Infeasible:
          self.__end = time.time()
          print("-------------------------------------------------------------------------------")
          print("Pase I has non-zero optimum : STOP - problem has not feasible solution!")
          print("-------------------------------------------------------------------------------")
          print("Execution statistics")
          self.PrintSolution()
          print("------------------------------------------------")
          print("LP is INFEASIBLE - no solution exists!")
          print("------------------------------------------------")
          sys.exit()
        else:
          self.PrintTableau("Initial feasible solution")
    else:
      self.OptStatus = OptStatus.Feasible
    self.__performPivots()
    # 
    self.__end = time.time()
    if self.OptStatus == OptStatus.NotBounded:
      print("-------------------------------------------------------------------------------")
      print("Found unbounded column => solution has no optimal solution")
      print("-------------------------------------------------------------------------------")
    print("Execution statistics")
    self.PrintSolution()
    if self.OptStatus == OptStatus.NotBounded:
      print("------------------------------------------------")
      print("LP is UNBOUNDED!")
      print("------------------------------------------------")



  def PrintTableau(self, header):
    print("------------------------------------------------")
    print("Simplex tableau: ", header, " Opt status: ", self.OptStatus)
    print("------------------------------------------------")
    # make sure to print all in one line - note that the width of the command prompt might wrap automatically
    # but output to file works just perfect!
    varNames = [self.__varName(item) for item in self.__basicVariables]
    varNames.append(self.__padStr("* OBJ *"))
    tmpArray = np.array([varNames]).T
    tableau = np.append(tmpArray, self.__tableau.copy(), axis=1)
    with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
      print(tableau)
    
  def ObjValue(self):
    objVal = self.__tableau[-1, -1]
    # phase I has no associated LP => we always have a min !
    if self.IsPhaseI:
      return -objVal
    return objVal if self.LP.ObjectiveType == OptimizationType.Max else -objVal

  def PrintSolution(self):
    if self.OptStatus == OptStatus.Optimal:
        if self.__PrintDetails:
            self.PrintTableau("FINAL TABLEAU")
        print("------------------------------------------------")
        print("Non-zero variables")
        nonZeros = []
        print(self.__basicVariables)
    
        for rowId, baseColId in enumerate(self.__basicVariables):
          # NOTE: in Phase 1, the LAST row is original objective => skip it:
          if (not self.IsPhaseI) & (baseColId < 0) & (rowId < len(self.__basicVariables) - 1 ):
            print("ERROR: basic variable with negative index at position ", rowId, " value " + baseColId)
            sys.exit()
          if baseColId >= 0:
            nonZeros.append([baseColId, "{} = {}".format(self.__varName(baseColId), "%.4f" % self.__getBasicVariableValue(rowId, baseColId))])
        # sort array
        nonZeros = sorted(nonZeros, key=lambda tup: tup[0])
        for val in nonZeros:
          print(val[1])
    else:
      print("------------------------------------------------")
      print("No solution found...")
    #
    print("------------------------------------------------")
    print("Num rows / cols     : ", self.NumRows, " / ", self.NumCols)
    print("Tableau dimensions: : ", self.__tableau.shape)
    print("Optimiser status    : ", self.OptStatus)
    print("Objective Value     : ", self.ObjValue())
    print("Nbr pivots Phase I  : ", self.__PhaseIPivotCount)
    print("Nbr pivots Phase II : ", self.__PivotCount)
    #print("Total exec time [s] : ", self.__end - self.__start)
    print("------------------------------------------------")

  def __padStr(self, str):
      return str.ljust(8, " ")

  def __varName(self, colId):
    if colId < 0:
      return "* OBJ_0 *"
    if colId < self.NumCols:
      return self.__padStr("x[{}]".format(colId))
    return self.__padStr("z[{}]".format(colId - self.NumCols))

  def __initTableau(self):
    print("-----entering __initTableau")
    # initializes the tableau from the previously loaded LP
    # tableau is
    # -------------
    # | A | I | b |
    # |-----------|
    # | c | 0 | 0 |
    # -------------
    # NOTE: to make things easier for Phase I and signs, always make sure b is >= 0 (if originally we have b < 0, simply change the signs of the entire row)
    # make sure to perform a deep copy of the coefficients - also make sure all are 2D arrays (i.d. add arrays as [])
    tmpA = self.LP.AMatrix.copy()
    tmpI = np.identity(self.NumRows)
    tmpB = np.array([self.LP.RHS.copy()]).T
    # now append an identity matrix to the right and the b vector
    self.__tableau = np.append(tmpA, tmpI, axis=1)
    self.__tableau = np.append(self.__tableau, tmpB, axis=1)
    # last row
    tmpC = np.array([np.concatenate((self.LP.Costs.copy(), np.zeros(self.NumRows + 1)))])
    self.__tableau = np.append(self.__tableau, tmpC, axis=0)
    self.__basicVariables = np.arange(self.NumCols, self.NumCols + self.NumRows, 1, dtype=int)
    self.TableauRowCount, self.TableauColCount = self.__tableau.shape
    for rowId in range(self.NumRows):
      if self.__tableau[rowId, -1] < -Constants.EPS:
        self.__tableau[rowId, :] = self.__tableau[rowId, :] * -1.
    if self.__PrintDetails:
      self.PrintTableau("Initial tableau")


  def __isSolutionFeasible(self):
    print("-----entering __isSolutionFeasible")
    # we MUST have that all BASIC variables are >= 0 to have a FEASIBLE solution - for that, check if there is a negative valued basic variable
    for rowId, baseColId in enumerate(self.__basicVariables):
        val = self.__getBasicVariableValue(rowId, baseColId)
        if val < -Constants.EPS:
          print("Current solution is not feasible: variable {} has value {}!".format(self.__varName(baseColId), self.__getBasicVariableValue(rowId, baseColId)))
          return False
    # all basic variables are >= 0, 
    return True

  def __getBasicVariableValue(self, rowId, baseColId):
      # the value of the basic variable is RHS / coeff in basic variable's colum
      return self.__tableau[rowId, -1] / self.__tableau[rowId, baseColId]

  def __solvePhaseI(self):   
    print("-----entering __solvePhaseI") 
    # initializes and solves the Phase I simplexe automatically
    phaseISplx = Simplexe()
    phaseISplx.initPhaseI(self)
    # if optimal solution of Phase I is NOT 0 => problem has no feasible solution
    if phaseISplx.ObjValue() > Constants.EPS:
      self.OptStatus = OptStatus.Infeasible
      return
    # make sure we eliminate ALL auxiliary variables from the current base (auxiliary vars are determined by their id : the all have id >= num cols + num rows)
    if self.__PrintDetails:
      phaseISplx.PrintTableau("Optimal solution Phase I")
    phaseISplx.RemoveAuxiliaryBasicVariables()
    phaseISplx.PrintSolution()
    # we have a feasible solution => extract feasible solution to proceed with Phase II
    # -> first part of the tableau (original matrix)
    self.__tableau = phaseISplx.__tableau[0:self.NumRows, 0:self.NumCols+self.NumRows]
    # -> append the objective row - previous last row of PhaseI tableau
    self.__tableau = np.append(self.__tableau, np.array([phaseISplx.__tableau[-2, 0:self.NumCols+self.NumRows]]), axis=0)
    # -> append the RHS for all except the last row
    tmp1 = np.array([phaseISplx.__tableau[0:-1, -1]]).T
    self.__tableau = np.append(self.__tableau, np.array([phaseISplx.__tableau[0:-1, -1]]).T, axis=1)
    # finally the basic variables indices
    self.__basicVariables = phaseISplx.__basicVariables[0:-1]
    self.__PhaseIPivotCount = phaseISplx.__PivotCount
    self.OptStatus = OptStatus.Feasible

  def initPhaseI(self, simplexe):
    print("-----entering initPhaseI")
    # create augmented tableau for Phase I => current
   	# -- Original -----		# -- Auxiliary -----
    # | A | I | b |			  | A | I | I | b |
    # |-----------|			  |---------------|
    # | c | 0 | 0 |		      | c | 0 | 0 | 0 | 
	# |-----------|			  |---------------|
    # 						  | d |-1 | 0 | S | 
	# d = - sum(coeff in column) and S is the -sum vector b (NOTE: we already made sure all elements in b are non-negative!)
    # store the fact we are actually using a Phase I - to make sure we don't select a pivot in the "original" objective row!
    self.IsPhaseI = True
    self.__start = time.time()
    # new dimensions (note : all original variables are considered as "normal" variables, whereas all auxiliary ones are now slacks!)
    self.NumRows = simplexe.NumRows + 1
    self.NumCols = simplexe.NumCols +  simplexe.NumRows
    self.OptStatus = OptStatus.Feasible
    # copy tableau excep LAST row and last column (we will add them last)
    tmpCopy = simplexe.__tableau.copy()
    self.__tableau = tmpCopy[0:-1,0:-1]
    # add the auxiliary variables and their objective row
    self.__tableau = np.append(self.__tableau, np.identity(simplexe.NumRows), axis=1)
    # add the RHS column
    self.__tableau = np.append(self.__tableau, np.array([tmpCopy[0:-1, -1]]).T, axis=1)
    # original and new "objective row" => initialize to zero - we will copy values individually
    # in BOTH rows, only first coefficients are non-zero - copy the original values (initial slack variables have 0 cost already => copy OK)
    # and for the NEW AUXILIARY objective row, the cost is -1 * the sum of column coefficients (here, objective is not yet added to the matrix => simply
    # take the sum is ok - also, for original slack variables, the sum is always -1 as we have an identity matrix)
    objRowOriginal = np.zeros(self.NumRows + self.NumCols , dtype=float)
    objRowNew = np.zeros(self.NumRows + self.NumCols , dtype=float)
    for iCol in range(self.NumCols):
        objRowOriginal[iCol] = tmpCopy[-1,iCol]
        objRowNew[iCol] = -np.sum(self.__tableau[:,iCol])
    # auxiliary objective is opposite sign (we have a minimization problem => objective value is POSITIVE => opposite sign is NEGATIVE)
    objRowNew[-1] = -np.sum(self.__tableau[:,-1])
    self.__tableau = np.append(self.__tableau, np.array([objRowOriginal]), axis=0)
    self.__tableau = np.append(self.__tableau, np.array([objRowNew]), axis=0)
    # current basic varialbes are all the new AUXILIARY variables - NOTE that the very LAST index is the one of the original objective function
    # => define its ID to -1 : if we use if anywhere, we will get an index out of range error!
    self.__basicVariables = np.arange(self.NumCols, self.NumCols + self.NumRows, 1, dtype=int)
    self.__basicVariables[-1] = -1
    # set dimensions 
    self.TableauRowCount, self.TableauColCount = self.__tableau.shape
    if self.__PrintDetails:
      self.PrintTableau("Phase 1 Tableau")
    self.__performPivots()

  def RemoveAuxiliaryBasicVariables(self):
    print("-----entering RemoveAuxiliaryBasicVariables")
    # after resolving PhaseI, we might still have some auxiliary varaibles in the current basis
    # if so, make sure we remove them - they always have index >= initial number of columns (i.e. self.NumCols here)
    maxId = np.argmax(self.__basicVariables)
    while (self.__basicVariables[maxId] >= self.NumCols):
      # leaving row is the one corresponding to the auxiliary variable, i.e. row with index maxId
      # we now have to retrieve which is the ORIGINAL SLACK COLUMN corresponding to the SAME row as the AUXILIARY variable (there is always one)
      # in Phase I, we have 1 additional row (corresponding to the original objective row) => the id of the original SLACK is id of AUXILIARY - (numRows - 1)
      originSlackVarId = self.__basicVariables[maxId] - (self.NumRows - 1)
      # pivot ROW ID is current index of auxiliary variable (i.e. maxId), and entering column is the original slack variable, i.e. originSlackVarId
      # - perform pivot
      self.__isRemovingAuxVariables = True
      self.__pivotTableau([maxId, originSlackVarId])
      self.__isRemovingAuxVariables = False
      # update the max Id
      maxId = np.argmax(self.__basicVariables)
    # stop measuring time 
    self.__end = time.time()

  def __performPivots(self):
    print("-----entering __performPivots")
    # we DO have a feasible solution - go on with the simplex
    while (self.OptStatus == OptStatus.Feasible):
      self.__pivotTableau(self.__selectPivot(PivotMode.MostNegative))
      #self.PrintTableau("After pivot")
      
  def __pivotTableau(self, pivotIDs):
    print("-----entering __pivotTableau")
    if pivotIDs is None or pivotIDs[0] < 0 or pivotIDs[1] < 0:
      # no pivot => optimiser status is updated in pivot selection => return!
      return
    # ---- update stats
    self.__PivotCount += 1
    #
    pivotRowId = pivotIDs[0]
    pivotColId = pivotIDs[1]
    #print("Pivot is ", pivotRowId, " ", pivotColId)
    pivotVal = self.__tableau[pivotRowId, pivotColId]
    # double-check pivot is actually positive
    if pivotVal < Constants.EPS:
      # pivoting around a negative value is allowed ONLY if we're doing it while removing auxiliary variables from PhaseI
      if not self.__isRemovingAuxVariables:
        print("ERROR: pivot on non-positive value {} at [{},{}]".format(pivotVal, pivotRowId, pivotColId))
        self.OptStatus = OptStatus.ERROR
        sys.exit()
    # single loop over full table - create the pivot row to be substracted from every row except the pivot row
    # for both performance (compute only once) and also to make sure we substract the correct values at every row
    # (if not, we might have an issue because of the pivot row being modified within the loop...)
    pivotRow = self.__tableau[pivotRowId, :] / pivotVal
    for rowId in range(self.TableauRowCount):
      # row of pivot is simply divided by pivot value
      if rowId == pivotRowId:
        self.__tableau[rowId, :] = pivotRow
      # any other row is the result of the row itself, minus the 
      else:
        self.__tableau[rowId, :] -= pivotRow * self.__tableau[rowId, pivotColId]
    # lastly, update the actual basic variable indices - replace the leaving index with the entering one
    self.__basicVariables[pivotRowId] = pivotColId

  # returns next pivot as an array of leaving row id and entering col Id - argument is the pivot mode (one of PivotMode enum)
  # return None if there is pivot (sets optimisation status accoringly before returning)
  def __selectPivot(self, pivotMode):  
    colId = self.__selectEnteringColumn(pivotMode)
    if (colId < 0):
      # no more entering column => optimiser status is OPTIMAL (must be - we never select a pivot if the tableau is unfeasible)
      self.OptStatus = OptStatus.Optimal
      return None
    # get leaving row id
    rowId = self.__selectLeavingRow(colId)
    if (rowId < 0):
      # there is a negative reduced cost column that never hits any constraint => problem is NotBounded!
      self.OptStatus = OptStatus.NotBounded
      return None
    # normal case : we do have a standard pivot!
    return [rowId, colId]

  # returns index of entering col Id depending on the chosen pivot mode passed as argument (one of PivotMode enum)
  # return -1 if there is pivot
  def __selectEnteringColumn(self, pivotMode):
    match pivotMode:
      case PivotMode.FirstNegative:
        # return index of first column that has negative reduced cost (i.e. negative value in LAST row) - except the LAST column
        return np.where(self.__tableau[-1, :-1] < -Constants.EPS)[0]

      case PivotMode.MostNegative:
        # return index of MOST negative column that has negative reduced cost (i.e. negative value in LAST row) - except the LAST column
        colId = self.__tableau[-1, :-1].argmin()
        return colId if self.__tableau[-1, colId] < -Constants.EPS else -1

      case PivotMode.MaxGain:
        # check the column that has most benefit i.e. maximal gain on objective (taking into account step calculation)
        print("NEG_TODO")

      case _:
        print("ERROR in selecting pivot: undefined pivot selection mode: ", pivotMode)
        sys.exit()
    

  # returns leaving row ID 
  # return None if there is pivot (sets optimisation status accoringly before returning)
  def __selectLeavingRow(self, pivotColId):
    minVal = None
    rowId = -1 
    # iterate over all columns except the last one and compute the ratio - make sure that in Phase 1 we also skip the original objective function!
    rowCount = self.TableauRowCount - 2 if self.IsPhaseI else self.TableauRowCount - 1
    for index in range(rowCount):
      val = self.__tableau[index, pivotColId]
      # make sure we update the index of argmin row only if the value is non-negative !
      if (val > Constants.EPS):
        if ((rowId < 0) or (self.__tableau[index, -1] / val < minVal)):
          rowId, minVal = index, self.__tableau[index, -1] / val
    return rowId if rowId >= 0 else -1


      