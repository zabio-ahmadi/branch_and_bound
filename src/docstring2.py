
class Constants:
  EPS = 1E-6

class OptimizationType(Enum):
   Min = 0
   Max = 1
   
class OptStatus(Enum):
   Unknown = 0
   Feasible = 1
   Infeasible = 2
   NotBounded = 3
   Optimal = 4
   ERROR = 99

class PivotMode(Enum):
   FirstNegative = 0
   MostNegative = 1
   MaxGain = 2

class Tableau:
  def __init__(self) -> None:
    """
    Initialize the Tableau object.
    
    Attributes:
      FileName (str): The name of the file containing the linear programming problem.
      Costs (np.array): A numpy array containing the coefficients of the objective function.
      AMatrix (np.array): A numpy array containing the coefficients of the constraint matrix.
      RHS (np.array): A numpy array containing the right-hand side values of the constraints.
      __tempA (List): A temporary list used to store the parsed constraint coefficients.
      __tempB (List): A temporary list used to store the parsed constraint right-hand side values.
      boolIsPhase1 (bool): A flag indicating whether the current phase is phase 1.
      boolIsRemovingAuxVariables (bool): A flag indicating whether auxiliary variables are being removed.
      PivotCount (int): The total number of pivots performed.
      Phase1PivotCount (int): The number of pivots performed in phase 1.
      boolPrintDetails (bool): A flag indicating whether to print detailed information about the tableau.
      OptStatus (OptStatus): The optimization status of the linear programming problem.
      NumRows (int): The number of rows in the tableau.
      NumCols (int): The number of columns in the tableau.
    """

  def loadFromFile(self, lpFileName: str, printDetails: bool) -> bool:
    """
    Load the linear programming problem from a file and parse its content.

    Args:
      lpFileName (str): The name of the file containing the linear programming problem.
      printDetails (bool): Whether to print detailed information about the tableau.

    Returns:
      bool: True if the file is successfully loaded and parsed, False otherwise.

    Notes:
      The file should contain the objective function, followed by the constraints, with one constraint per line.
      Coefficients, signs, and right-hand side values should be separated by semicolons.
    """


  def initTableau(self) -> None:
    """
    Initialize the tableau from the previously loaded linear programming problem.

    Steps:
      1. Create a copy of the AMatrix.
      2. Create an identity matrix of size NumRows.
      3. Create a column vector from the RHS values.
      4. Append the identity matrix and RHS column vector to the AMatrix copy.
      5. Create a row vector with the Costs and zeros, and append it to the tableau.
      6. Update the basic variables.
      7. Ensure all right-hand side values in the tableau are non-negative.
      8. Print the initial tableau if boolPrintDetails is set to True.

    Notes:
      The initialized tableau has the following structure:
      -------------
      | A | I | b |
      |-----------|
      | c | 0 | 0 |
      -------------
    """


  def solveProblem(self) -> None:
    """
    Solve the linear programming problem.

    Steps:
      1. Check if the initial solution is feasible.
      2. If not feasible, execute Phase 1 to obtain a feasible solution.
      3. Perform pivots to optimize the solution.
      4. Print the execution statistics.

    Raises:
      SystemExit: If the problem has no feasible or optimal solution, the function
            will terminate the program with an error message.
    """


  def ObjValue(self) -> float:
    """
    Calculate the objective value of the solution.

    Returns:
      float: The objective value of the solution.

    Notes:
      If the problem is in Phase 1, the objective value is negated.
      If the problem is a maximization problem, the objective value is returned as-is.
      If the problem is a minimization problem, the objective value is negated.
    """

  def isSolutionFeasible(self) -> bool:
    """
    Check if the current solution is feasible.

    Returns:
      bool: True if the solution is feasible, False otherwise.

    Notes:
      A solution is considered feasible if all basic variables are greater than or equal
      to 0 (with a tolerance of Constants.EPS).
    """


  def getBasicVariableValue(self, rowId: int, baseColId: int) -> float:
    return self.tableau[rowId, -1] / self.tableau[rowId, baseColId]

  def solvePhase1(self) -> None:
    """
    Initializes and solves the Phase 1 simplex automatically.

    Steps:
    1. Initialize the Phase 1 simplex tableau.
    2. Check if the optimal solution of Phase 1 is greater than the specified epsilon.
    3. If the optimal solution is greater than epsilon, the problem has no feasible solution.
    4. Remove all auxiliary basic variables from the current base.
    5. Extract the feasible solution to proceed with Phase 1.

    Notes:
      The OptStatus of the Tableau object will be updated to Infeasible or Feasible based on the results of this method.
    """

  def initPhase1(self, simplexe: 'Tableau') -> None:
    """
    Initializes the Phase 1 simplex tableau.

    Steps:
    1. Create an augmented tableau for Phase 1.
    2. Update the dimensions of the tableau.
    3. Copy the original tableau except for the last row and column.
    4. Add the auxiliary variables and their objective row.
    5. Add the RHS column.
    6. Update the objective rows.
    7. Add the current basic variables.
    8. Set the dimensions of the tableau.

    Notes:
      The OptStatus of the Tableau object will be updated to Feasible.
      # create augmented tableau for Phase 1 => current
      # -- Original -----        # -- Auxiliary -----
      # | A | I | b |              | A | I | I | b |
      # |-----------|              |---------------|
      # | c | 0 | 0 |              | c | 0 | 0 | 0 | 
      # |-----------|              |---------------|
      #                            | d |-1 | 0 | S | 
      # d = - sum(coeff in column) and S is the -sum vector b (we already made sure all elements in b are non-negative!)

    """


  def RemoveAuxiliaryBasicVariables(self):
    """
    Removes any remaining auxiliary basic variables from the current basis after completing Phase 1 of the two-phase simplex method.
    This function uses the `pivotTableauOnce` method to perform the necessary pivoting operations.
    """

  def performPivots(self) -> None:
    """
    Executes simplex algorithm on the tableau to find an optimal solution.
    
    Steps:
      1. Check if the current solution is feasible.
      2. Perform the simplex algorithm by repeatedly selecting and applying pivots.
      3. Update the optimization status after each pivot.
      4. Terminate when an optimal solution is found or an error occurs.
    """

  def pivotTableauOnce(self, pivotIDs: Optional[Tuple[int, int]]) -> None:
    """
    Performs a single pivot operation on the tableau.
    
    Args:
      pivotIDs (Optional[Tuple[int, int]]): A tuple with the row and column indices of the pivot element.
        If no pivot is provided or any index is negative, the function returns without performing any operation.
    
    Steps:
      1. Check if the pivot is valid.
      2. Perform the pivot operation on the tableau.
      3. Update the basic variables accordingly.
    """

