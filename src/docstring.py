class LP:
 """
 This class represents a linear programming problem.

 Attributes:
 -----------
 FileName (str): The name of the file containing the linear programming problem.
 Costs (numpy.ndarray): An array of coefficients representing the objective function.
 AMatrix (numpy.ndarray): A matrix of coefficients representing the constraints.
 RHS (numpy.ndarray): An array of values representing the right-hand side of the constraints.
 __tempA (List[List[float]]): A temporary list used to store the matrix of coefficients representing the constraints.
 __tempB (List[float]): A temporary list used to store the values representing the right-hand side of the constraints.

 Methods:
 --------
 __init__(self, lpFileName: str) -> None
  Initializes the LP class with the name of the file containing the linear programming problem.
 ParseFile() -> bool
  Reads the file and parses the linear programming problem.
  Updates: self.AMatrix, self.Costs, and self.RHS.
 parseObjective(sLine: str) -> None
  Parses the objective function.
  Updates: self.ObjectiveType and self.Costs.
 parseConstraint(sLine: str) -> None
  Parses a constraint.
  Updates: self.__tempA and self.__tempB.
 PrintProblem(boolPrint: bool = False) -> None
  Prints the problem.
 """

class Simplexe:
 """
 The Simplexe class is an implementation of the Simplex algorithm for solving Linear Programming (LP) problems.
 
 Attributes:
  IsPhaseI (bool): Indicates if the algorithm is in Phase I.
  __isRemovingAuxVariables (bool): Indicates if the algorithm is removing auxiliary variables.
  __PivotCount (int): The number of pivot operations performed.
  __PhaseIPivotCount (int): The number of pivot operations performed in Phase I.
  __PrintDetails (bool): Indicates if the algorithm should print detailed information.
  __basicVars (np.ndarray): An array of basic variable indices.
  OptStatus (int): The current status of the optimization process.
  LP (LP): An instance of the LP class, representing the linear programming problem.
  NumRows (int): The number of rows in the LP problem.
  NumCols (int): The number of columns in the LP problem.
  printDetails (bool): Indicates whether to print the details of the optimization process or not.

 Methods: 
  __init__(self) -> None:
   Initializes the Simplexe class attributes.

  LoadFromFile(self, lpFileName: str, printDetails: bool) -> None:
   Loads a linear programming problem from a file and solves it.
   Updates: self.FileName, self.__PrintDetails, self.printDetails, self.LP, self.OptStatus, self.NumRows, self.NumCols

  __solveProblem(self) -> None:
   Solves the LP problem using the Simplex algorithm.
   Updates: self.__start, self.__end, self.OptStatus

  __performPivots(self) -> None:
   Performs pivot operations until the problem is solved.
   Updates: self.OptStatus

  PrintTab(self, header: str) -> None:
   Prints the current Simplex tableau with a given header.

  ObjValue(self) -> float:
   Returns the objective value of the current solution.

  PrintSolution(self, boolPrint: bool) -> None:
   Prints the current solution and optimization statistics.

  __padStr(self, str: str) -> str:
   Pads a string with spaces up to a length of 8.

  __varName(self, colId: int) -> str:
   Returns the variable name for a given column index.

  __initTab(self) -> None:
   Initializes the Simplex tableau.
   Updates: self.__tab, self.__basicVars, self.TabRowCount, self.TabColCount

  __isSolutionFeasible(self) -> bool:
   Checks if the current solution is feasible.

  __getBasicVariableValue(self, rowId: int, baseColId: int) -> float:
   Returns the value of a basic variable in the tableau.

  __solvePhaseI(self) -> None:
   Solves Phase I of the Simplex algorithm.
   Updates: self.__end, self.OptStatus

  initPhaseI(self, simplexe: Simplexe) -> None:
   Initializes the Simplexe object for Phase I.
   Updates: self.IsPhaseI, self.__start, self.NumRows, self.NumCols, self.OptStatus, self.__tab, self.__basicVars, self.TabRowCount, self.TabColCount

  RemoveAuxiliaryBasicVariables(self) -> None:
   Removes auxiliary basic variables from the tableau.
   Updates: self.__end

  __pivotTab(self, pivotIDs: List[int]) -> None:
   Pivots the Simplex tableau using the specified row and column indices.
   Updates: self.__PivotCount

  __selectPivot(self, pivotMode: PivotMode) -> Union[None, List[int]]:
   Selects the pivot row and column based on the given pivot mode.

  __selectEnteringColumn(self, pivotMode: PivotMode) -> int:
   Selects the entering column based on the given pivot mode.

  __selectLeavingRow(pivotColId: int) -> int: 
   Selects the leaving row for a given entering column.
 """