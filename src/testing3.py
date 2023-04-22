import numpy as np
from typing import *
import warnings

class Tab:
    """
    Simplex Tableau implementation with the following attributes:
    - c: np.ndarray[float]: The coefficients of the objective function, 1D array. Setter calls update_reduced_costs().
    - A: np.ndarray[float]: The coefficients of the constraints, 2D array. Setter calls update_tableau().
    - b: np.ndarray[float]: The right-hand side of the constraints, 1D array. Setter calls update_basic_var_values().
    - num_vars: int: The number of variables in the linear programming problem.
    - num_constraints: int: The number of constraints in the linear programming problem.
    - reduced_costs: np.ndarray[float]: The reduced costs of the variables, 1D array.
    - basic_vars: np.ndarray[int]: The indices of the variables in the current basis, 1D array.
    - basic_var_values: np.ndarray[float]: The values of the basic variables in the current solution, 1D array.
    - artificial_vars: List[int]: The indices of the artificial variables introduced in the auxiliary problem.
    - artificial_var_values: np.ndarray[float]: The values of the artificial variables in the current solution, 1D array.
    - slack_vars: np.ndarray[float]: The values of the slack variables, 1D array.
    - phase: int: The current phase of the simplex algorithm (1 or 2).
    
    Properties with getter and setter methods:
    - c
    - A
    - b
    - num_vars
    - num_constraints
    - reduced_costs
    - basic_vars
    - basic_var_values
    - artificial_var_values
    - phase
    
    Methods called when an attribute is set:
    - update_reduced_costs() (called when c is set)
    - update_tableau() (called when A is set or when num_vars or num_constraints is set)
    - update_basic_var_values() (called when b is set)
    
    Additional methods:
    - check_optimality() -> bool: Checks the optimality of the current solution.
    
    Methods called when an attribute is read:
    - none
    """

    def __init__(self, c: np.ndarray[float] = None,
                 A: np.ndarray[float] = None,
                 b: np.ndarray[float] = None) -> None:

        self._c: np.ndarray[float] = None
        self._c_full = None
        self._A: np.ndarray[float] = None
        self._b: np.ndarray[float] = None
        self._num_vars: int = None
        self._num_constraints: int = None
        self._reduced_costs: np.ndarray[float] = None
        self._basic_vars: np.ndarray[int] = None
        self._basic_var_values: np.ndarray[float] = None
        self._artificial_vars: List[int] = []
        self._artificial_var_values: np.ndarray[float] = None
        self._slack_vars: np.ndarray[float] = None
        self._phase: int = 2

        if c is not None and A is not None and b is not None:
            self._num_vars: int = c.shape[0]
            self._num_constraints: int = A.shape[0]

            # Check if dimensions are consistent
            if A.shape[1] != self._num_vars or b.shape[0] != self._num_constraints:
                raise ValueError("Inconsistent dimensions for c, A, and b.")

            # Concatenate identity matrix to represent slack variables
            A_with_slack_vars = np.hstack([A, np.eye(self._num_constraints)])

            self._basic_vars: np.ndarray[int] = None if self._num_vars is None or self._num_constraints is None else np.arange(self._num_vars, self._num_vars + self._num_constraints, dtype=int)
            self._c: np.ndarray[float] = c
            self._A: np.ndarray[float] = A_with_slack_vars
            self._b: np.ndarray[float] = b

            # Create _c_full with the coefficients for all variables
            self._c_full = np.concatenate((c, np.zeros(self._num_constraints)))

            self.update_basic_var_values()
            self.update_reduced_costs()
        else:
            self._num_vars: int = None
            self._num_constraints: int = None

    @property
    def c(self) -> np.ndarray[float]:
        return self._c

    @c.setter
    def c(self, value: np.ndarray[float]) -> None:
        self._c = value
        self.update_reduced_costs()

    @property
    def A(self) -> np.ndarray[float]:
        return self._A

    @A.setter
    def A(self, value: np.ndarray[float]) -> None:
        self._A = value
        self.update_tableau()

    @property
    def b(self) -> np.ndarray[float]:
        return self._b

    @b.setter
    def b(self, value: np.ndarray[float]) -> None:
        self._b = value
        self.update_basic_var_values()

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def num_constraints(self) -> int:
        return self._num_constraints

    @property
    def reduced_costs(self) -> np.ndarray[float]:
        return self._reduced_costs

    @property
    def basic_vars(self) -> np.ndarray[int]:
        return self._basic_vars

    @property
    def basic_var_values(self) -> np.ndarray[float]:
        return self._basic_var_values

    @property
    def artificial_var_values(self) -> np.ndarray[float]:
        return self._artificial_var_values

    @property
    def phase(self) -> int:
        return self._phase

    @phase.setter
    def phase(self, value: int) -> None:
        self._phase = value

    def update_reduced_costs(self) -> None:
        if self._c is not None and self._A is not None and self._basic_vars is not None:
            reduced_costs = np.zeros(self._num_vars)
            for j in range(self._num_vars):
                if j not in self._basic_vars:
                    reduced_costs[j] = self._c[j] - np.sum(self._A[:, j] * self._c_full[self._basic_vars])

            self._reduced_costs = reduced_costs

    def update_tableau(self) -> None:
        if self._A is not None and self._num_constraints is not None:
            A_with_slack_vars = np.hstack([self._A[:, :self._num_vars], np.eye(self._num_constraints)])
            self._A = A_with_slack_vars

            # Update reduced costs after updating the tableau
            self.update_reduced_costs()

    def update_basic_var_values(self) -> None:
        if self._A is not None and self._b is not None and self._basic_vars is not None:
            B = self._A[:, self._basic_vars]
            self._basic_var_values = np.linalg.solve(B, self._b)

    def check_optimality(self) -> bool:
        if self._reduced_costs is not None:
            return np.all(self._reduced_costs >= 0)
        else:
            raise ValueError("Reduced costs have not been initialized.")
    
    def is_canonical(self) -> bool:
        """
        Returns True if the tableau is in canonical form.
        """
        # The tableau is in canonical form if the basic variables are the first num_constraints columns
        return self.basic_vars == list(range(self.num_constraints))

    def to_canonical_form(self) -> None:
        """
        Converts the tableau to canonical form.
        """
        # Find a set of basic variables that form an identity matrix in the constraints matrix A
        I = np.eye(self.num_constraints)
        for i in range(self.num_vars):
            if i not in self.basic_vars:
                # Check if the ith column of A is an element of the identity matrix I
                mask = np.isclose(self.A[:, i], I[:, 0], rtol=1e-10, atol=1e-10)
                for j in range(1, self.num_constraints):
                    if not np.allclose(self.A[:, i][mask], I[:, j][mask], rtol=1e-10, atol=1e-10):
                        # The ith column of A is not part of any identity matrix in A
                        break
                else:
                    # The ith column of A is part of an identity matrix in A
                    basic_var = i
                    break
        else:
            # No set of basic variables that form an identity matrix in A was found
            raise ValueError('Tableau is not in canonical form')

        # Perform pivot operations to bring the tableau to canonical form
        while not self.is_canonical():
            # Choose the leaving variable
            leaving_row = self._choose_leaving_variable(basic_var)

            # Choose the entering variable
            entering_col = self._choose_entering_variable(leaving_row)

            # Perform the pivot operation
            self._pivot(leaving_row, entering_col)

            # Update the basic and nonbasic variable lists
            self.basic_vars[leaving_row] = entering_col
            self.nonbasic_vars.remove(entering_col)
            self.nonbasic_vars.append(basic_var)

            # Choose the next basic variable
            for i in range(self.num_constraints):
                if self.basic_vars[i] == basic_var:
                    self.basic_vars[i] = entering_col
                    break

            # Update the basic variable values and reduced costs
            self.basic_var_values[leaving_row] = self.b[leaving_row] / self.A[leaving_row, entering_col]
            self.reduced_costs[self.nonbasic_vars] = self.c[self.nonbasic_vars] - np.dot(self.c[self.basic_vars], self.A[:, self.nonbasic_vars])
            self.reduced_costs[self.basic_vars] = 0

            # Choose the next basic variable
            basic_var = self.nonbasic_vars.pop(0)

        # Remove artificial variables and update reduced costs
        if self._artificial_vars is not None:
            self.A = np.delete(self.A, self._artificial_vars, axis=1)
            self.c = np.delete(self.c, self._artificial_vars)
            self.nonbasic_vars = [var for var in self.nonbasic_vars if var not in self._artificial_vars]
            self.reduced_costs = self.c - np.dot(self.c[self.basic_vars], self.A)

        # Update the phase
        self.phase = 2


class Simplex:
    """
    Simplex solver for linear programming problems.
    
    Attributes:
    - tab: Tab: An instance of the Tab class representing the simplex tableau.
    - optimal_solution: np.ndarray[float]: The optimal solution of the linear programming problem, 1D array.
    - optimal_value: float: The optimal value of the objective function.
    - status: str: The status of the solution ('optimal', 'infeasible', 'unbounded', or 'not solved').
    
    Methods:
    - __init__(self, c: Optional[np.ndarray[float]] = None, A: Optional[np.ndarray[float]] = None, b: Optional[np.ndarray[float]] = None, max_iter: int = 1000): Initializes the solver with the given LP problem or an empty instance.
    - loadFromFile(self, filepath: str) -> None: Loads an LP problem from a file and updates the tableau.
    - solve(self) -> str: Solves the LP problem and returns the solution status.
    - _initialize_simplex(self) -> None: Initializes the tableau and phase 1 of the algorithm.
    - _phase_one(self) -> None: Executes phase 1 of the simplex algorithm.
    - _phase_two(self) -> None: Executes phase 2 of the simplex algorithm.
    - _pivot(self, row: int, col: int) -> None: Performs a pivot operation on the tableau.
    - _choose_entering_variable(self) -> int: Determines the entering variable using the most negative reduced cost rule.
    - _choose_leaving_variable(self, col: int) -> int: Determines the leaving variable using the minimum ratio test.
    - _is_optimal(self) -> bool: Checks if the current solution is optimal.

    Additional methods for working with branch and bound:
    - fractional_variables(self) -> List[int]: Returns a list of indices of fractional variables in the current solution.
    - current_solution(self) -> np.ndarray[float]: Returns the current solution of the LP problem.
    - current_objective_value(self) -> float: Returns the current value of the objective function.
    """

    def __init__(self, c: Optional[np.ndarray[float]] = None,
                 A: Optional[np.ndarray[float]] = None,
                 b: Optional[np.ndarray[float]] = None,
                 max_iter: int = 1000) -> None:
        if c is None or A is None or b is None:
            self.tab = None
        else:
            self.tab = Tab(c, A, b)

        self.optimal_solution = None
        self.optimal_value = None
        self.status = 'not solved'
        self.max_iter = max_iter

    def reset(self) -> None:
        self.optimal_solution = None
        self.optimal_value = None
        self.status = 'not solved'

    def solve(self) -> str:
        self.reset()
        self._initialize_simplex()
        self._phase_one()
        if self.status == 'infeasible':
            return self.status

        self._phase_two()
        if self.status == 'optimal':
            self.optimal_solution = self.tab.basic_var_values
            self.optimal_value = self.tab.c @ self.optimal_solution
        elif self.status == 'unbounded':
            return self.status

        return self.status


    def _initialize_simplex(self) -> None:
        if np.any(self.tab.b < 0):
            # If any of the constraint right-hand side values are negative, artificial variables are needed
            artificial_vars = np.eye(self.tab.num_constraints)
            self.tab.A = np.hstack([self.tab.A, artificial_vars])
            self.tab._artificial_vars = list(range(self.tab.num_vars, self.tab.num_vars + self.tab.num_constraints))
            self.tab.phase = 1
        else:
            # If all right-hand side values are non-negative, make sure the tableau is in canonical form
            if not self.tab.is_canonical():
                self.tab.to_canonical_form()
            if np.any(self.tab.basic_var_values < 0):
                # If any basic variable is negative, pivot to make it non-negative
                entering_col = np.argmin(self.tab.basic_var_values)
                leaving_row = self._choose_leaving_variable(entering_col)
                self._pivot(leaving_row, entering_col)
            self.tab.phase = 2


    def _phase_one(self) -> None:
        if self.tab.phase == 1:
            artificial_c = np.zeros(self.tab.num_vars + self.tab.num_constraints)
            artificial_c[self.tab._artificial_vars] = 1
            original_c = self.tab.c.copy()
            self.tab.c = artificial_c

            num_iter = 0
            while not self.tab.check_optimality() and num_iter < self.max_iter:
                entering_col = self._choose_entering_variable()
                leaving_row = self._choose_leaving_variable(entering_col)

                if leaving_row is None:
                    self.status = 'unbounded'
                    return

                self._pivot(leaving_row, entering_col)
                num_iter += 1

            if np.isclose(self.tab.basic_var_values[self.tab._artificial_vars], 0).any():
                self.status = 'infeasible'
                return

            # Remove artificial variables from the tableau and restore the original objective function
            self.tab.A = np.delete(self.tab.A, self.tab.artificial_vars, axis=1)
            self.tab.c = original_c
            self.tab.update_reduced_costs()

    def _phase_two(self) -> None:
        num_iter = 0
        while not self._is_optimal() and num_iter < self.max_iter:
            entering_col = self._choose_entering_variable()
            leaving_row = self._choose_leaving_variable(entering_col)

            if leaving_row is None:
                self.status = 'unbounded'
                return

            self._pivot(leaving_row, entering_col)
            num_iter += 1

        if self._is_optimal():
            self.status = 'optimal'
        else:
            self.status = 'not solved'


    def _is_optimal(self) -> bool:
        basic_reduced_costs = self.tab.reduced_costs[:self.tab.num_vars]
        non_basic_reduced_costs = self.tab.reduced_costs[self.tab.num_vars:]
        if np.any(non_basic_reduced_costs < 0):
            # There may be alternate optimal solutions
            return False
        return np.all(basic_reduced_costs >= 0)


    def loadFromFile(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse objective type and coefficients
        objective_line = lines[0].strip().split(';')
        objective_type = objective_line[0].lower()
        c = np.array([float(coef) for coef in objective_line[1:-1]])
        if objective_type == 'max':
            c = -c

        # Parse constraints
        A = []
        b = []
        for constraint_line in lines[1:]:
            constraint_parts = constraint_line.strip().split(';')
            constraint_coeffs = [float(coef) for coef in constraint_parts[:-3]]
            constraint_type = constraint_parts[-3].strip().lower()
            constraint_rhs = float(constraint_parts[-2])

            # Force all constraints to be of the form "<="
            if constraint_type == '>=':
                constraint_coeffs = [-coef for coef in constraint_coeffs]
                constraint_rhs = -constraint_rhs
            elif constraint_type == '=':
                # Convert "=" constraint to two "<=" constraints
                A.append(constraint_coeffs)
                b.append(constraint_rhs)
                constraint_coeffs = [-coef for coef in constraint_coeffs]
                constraint_rhs = -constraint_rhs

            A.append(constraint_coeffs)
            b.append(constraint_rhs)

        A = np.array(A)
        b = np.array(b)

        # Force the problem to be of "min" type
        if objective_type == 'max':
            A = -A
            b = -b
            c = -c

        # Set the attributes
        self.tab = Tab(c, A, b)


class BranchAndBound:
    """
    Branch and Bound solver for integer linear programming problems.
    
    Attributes:
    - c: np.ndarray[float]: The coefficients of the objective function, 1D array.
    - A: np.ndarray[float]: The coefficients of the constraints, 2D array.
    - b: np.ndarray[float]: The right-hand side of the constraints, 1D array.
    - max_iter: int: The maximum number of iterations to perform.
    - optimal_solution: np.ndarray[float]: The optimal solution of the ILP problem, 1D array.
    - optimal_value: float: The optimal value of the objective function.
    - status: str: The status of the solution ('optimal', 'infeasible', 'unbounded', or 'not solved').
    
    Methods:
    - __init__(self, c: np.ndarray[float], A: np.ndarray[float], b: np.ndarray[float], max_iter: int = 1000): Initializes the solver with the given ILP problem.
    - solve(self) -> str: Solves the ILP problem and returns the solution status.
    - _branch(self, node: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]: Creates two branches from the given node by adding constraints on a fractional variable.
    - _bound(self, node: Dict[str, Any]) -> float: Solves the LP relaxation of the given node using the Simplex class and returns the objective value.
    - _is_integer_solution(self, solution: np.ndarray[float]) -> bool: Checks if the given solution is an integer solution.
    """



simplex = Simplex()
simplex.loadFromFile('/home/stefan/Nextcloud/school/2022-2023/printemps/Math/math-plne/src/lp_glaces.txt')
simplex.solve()
print(simplex)

