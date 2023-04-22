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
    
    Methods called when an attribute is set:
    
    - update_reduced_costs() (called when c is set)
    - update_tableau() (called when A is set or when num_vars or num_constraints is set)
    - update_basic_var_values() (called when b is set)
    
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
    def c(self) -> Union[None, np.ndarray[float]]:
        """
        The coefficients of the objective function, 1D array.

        Returns:
            np.ndarray[float]: The coefficients of the objective function.

        Raises:
            ValueError: If c is not a 1D numpy array or if its length is not equal to num_vars.
            TypeError: If c is not a numpy array.
        """
        if self._c is None:
            return None
        elif not isinstance(self._c, np.ndarray):
            raise TypeError("c must be a numpy array.")
        elif self._c.ndim != 1:
            raise ValueError("c must be a 1D numpy array.")
        elif self._c.shape[0] != self._num_vars:
            raise ValueError("c must have the same length as the number of variables.")
        else:
            return self._c

    @c.setter
    def c(self, value: np.ndarray[float]) -> None:
        """
        Setter method for the c attribute.
        
        Parameters:
        - value: 1D numpy array[float]: The new coefficients of the objective function.
        
        Raises:
        - ValueError: If the value parameter is not a 1D numpy array of floats.
        - ValueError: If the length of the value parameter does not match the number of variables in the tableau.
        In this case, the user should modify the num_vars attribute so that the tableau is extended or reduced accordingly.
        """
        if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype.kind != 'f':
            raise ValueError("The c attribute must be set to a 1D numpy array of floats.")
        if self._num_vars is not None and value.shape[0] != self._num_vars:
            raise ValueError(f"The length of the c array must match the number of variables in the tableau ({self._num_vars}). "
                            "Please modify the num_vars attribute to extend or reduce the tableau accordingly.")
        self._c = value
        self.update_reduced_costs()

    @property
    def A(self) -> np.ndarray[float]:
        """
        Getter for the A attribute.

        Returns:
        - np.ndarray[float]: The coefficients of the constraints, 2D array.

        Raises:
        - ValueError: If A is not a 2D numpy array.
        - ValueError: If the number of rows in A does not match the number of constraints.
        - ValueError: If the number of columns in A does not match the number of variables.
        """
        if self._A is None:
            return None

        if not isinstance(self._A, np.ndarray):
            raise ValueError("A must be a numpy array.")

        if len(self._A.shape) != 2:
            raise ValueError("A must be a 2D array.")

        if self._num_constraints is not None and self._A.shape[0] != self._num_constraints:
            raise ValueError(f"Number of rows in A ({self._A.shape[0]}) must match number of constraints ({self._num_constraints}).")

        if self._num_vars is not None and self._A.shape[1] != self._num_vars:
            raise ValueError(f"Number of columns in A ({self._A.shape[1]}) must match number of variables ({self._num_vars}).")

        return self._A.copy()


    @A.setter
    def A(self, value: np.ndarray[float]) -> None:
        """
        Setter method for the A attribute. Calls update_tableau().

        Parameters:
        - value: np.ndarray[float]: The new value of the A attribute.

        Raises:
        - ValueError: If the input is not a 2D numpy array or if the number of columns does not match the number of variables.
        """
        if value is None:
            self._A = None
            self._num_constraints = None
            return

        if not isinstance(value, np.ndarray) or value.ndim != 2:
            raise ValueError("A must be a 2D numpy array.")

        num_rows, num_cols = value.shape
        if num_cols != self._num_vars:
            raise ValueError(f"The number of columns in A ({num_cols}) must match the number of variables ({self._num_vars}).")

        self._A = value
        self._num_constraints = num_rows
        self.update_tableau()

    @property
    def b(self) -> np.ndarray[float]:
        """
        The right-hand side of the constraints, 1D array.

        Returns:
        np.ndarray[float]: The right-hand side of the constraints.

        Raises:
        ValueError: If b is not set or is not a 1D numpy array.
        ValueError: If the length of b is not equal to the number of constraints in the tableau.
        """
        if self._b is None:
            raise ValueError("The b attribute is not set. Please set it before accessing it.")
        if not isinstance(self._b, np.ndarray) or self._b.ndim != 1:
            raise ValueError("The b attribute must be a 1D numpy array.")
        if len(self._b) != self._num_constraints:
            raise ValueError(f"The length of b ({len(self._b)}) must be equal to the number of constraints "
                            f"in the tableau ({self._num_constraints}).")
        if (self._b < 0).any():
            warnings.warn("The b attribute should ideally be non-negative. Please ensure that all elements "
                        "of b are non-negative if possible.")
        return self._b

    @b.setter
    def b(self, value: np.ndarray[float]) -> None:
        """Setter calls update_basic_var_values()."""
        if value is None:
            raise ValueError("b cannot be None")
        if not isinstance(value, np.ndarray):
            raise TypeError("b must be a numpy array")
        if value.ndim != 1:
            raise ValueError("b must be a 1D array")
        if len(value) != self.num_constraints:
            raise ValueError(f"b must have length {self.num_constraints}")
        if np.any(np.isnan(value)):
            raise ValueError("b cannot contain NaN values")
        if np.any(np.isinf(value)):
            raise ValueError("b cannot contain infinite values")
        if np.any(value < 0):
            warnings.warn("b contains negative values, which may cause issues with branch and bound algorithms")
        self._b = value
        self.update_basic_var_values()


    @property
    def num_vars(self) -> int:
        """
        Getter for the num_vars attribute.

        Returns:
        - int: The number of variables in the linear programming problem.

        Raises:
        - ValueError: If num_vars is not a positive integer.
        - ValueError: If num_vars does not match the number of columns in A.
        """
        if self._num_vars is None:
            raise ValueError("num_vars has not been set.")
        if not isinstance(self._num_vars, int) or self._num_vars <= 0:
            raise ValueError("num_vars must be a positive integer.")
        if self._A is not None and self._num_vars != self._A.shape[1]:
            raise ValueError("num_vars does not match the number of columns in A.")
        return self._num_vars


    @num_vars.setter
    def num_vars(self, value: int) -> None:
        """
        Sets the number of variables in the linear programming problem and updates the tableau.

        Args:
            value: int: The new number of variables.

        Raises:
            ValueError: If value is not a positive integer.
            ValueError: If the number of constraints and the number of variables are not both positive integers.
            ValueError: If the number of variables is less than the number of basic variables in the current tableau.
            Warning: If the number of variables is greater than the number of columns in the current tableau.

        Returns:
            None.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("num_vars must be a positive integer")

        if self.num_constraints is None or self.num_constraints <= 0:
            raise ValueError("Both num_vars and num_constraints must be positive integers")

        if value < self.basic_vars.size:
            raise ValueError("num_vars must be greater than or equal to the number of basic variables")

        new_A_shape = (self.num_constraints, value + len(self._artificial_vars))

        if new_A_shape[1] > self.tableau.shape[1]:
            message = "Warning: increasing num_vars will introduce new non-basic variables with value 0."
            message += " Consider adding them manually before running simplex algorithm."
            warnings.warn(message)

        new_A = np.zeros(new_A_shape)

        if self.A is not None:
            if value > self._num_vars:
                new_A[:, :-len(self._artificial_vars)] = self.A[:, :-len(self._artificial_vars)]
            else:
                new_A = self.A[:, :-len(self._artificial_vars)]

        self._num_vars = value
        self._A = new_A
        self.update_tableau()


    @property
    def num_constraints(self) -> int:
        """
        The number of constraints in the linear programming problem.

        Raises:
        - ValueError: If _A or _b is None.
        - ValueError: If _A and _b do not have compatible shapes.
        - ValueError: If any element of _b is negative.

        Returns:
        - int: The number of constraints in the linear programming problem.
        """
        if self._A is None or self._b is None:
            raise ValueError("The _A and _b matrices must be set before num_constraints can be computed.")
        if self._A.shape[0] != self._b.shape[0]:
            raise ValueError("The _A and _b matrices must have the same number of rows.")
        if np.any(self._b < 0):
            raise ValueError("All elements of _b must be non-negative.")
        return self._b.shape[0]


    @num_constraints.setter
    def num_constraints(self, value: int) -> None:
        """Updates the num_constraints attribute and the size of A and b accordingly.

        Args:
            value (int): The new number of constraints.

        Raises:
            ValueError: If the value is not a positive integer.
            ValueError: If the value is smaller than the number of basic variables in the current tableau.
            ValueError: If the value is smaller than the number of artificial variables in the current tableau.
            Warning: If the value is larger than the number of variables in the current tableau.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("num_constraints must be a positive integer")

        if self.basic_vars is not None and len(self.basic_vars) > value:
            raise ValueError("num_constraints cannot be smaller than the number of basic variables")

        if len(self._artificial_vars) > value:
            raise ValueError("num_constraints cannot be smaller than the number of artificial variables")

        if self._num_vars is not None and value > self._num_vars:
            warning_msg = "Warning: increasing num_constraints beyond the current number of variables " \
                          "will result in redundant rows in the tableau. Consider setting num_vars to a " \
                          "larger value instead."
            warnings.warn(warning_msg)

        # Update num_constraints attribute
        self._num_constraints = value

        # Resize A and b if necessary
        if self.A is not None and self.b is not None:
            if self.A.shape[0] < value:
                zeros_to_add = np.zeros((value - self.A.shape[0], self._num_vars))
                self._A = np.vstack([self.A, zeros_to_add])
                self._b = np.concatenate([self.b, np.zeros(value - self.A.shape[0])])
            elif self.A.shape[0] > value:
                self._A = self.A[:value, :]
                self._b = self.b[:value]

        # Update tableau if necessary
        if self.A is not None and self._num_vars is not None:
            self.update_tableau()


    @property
    def reduced_costs(self) -> Union[None, np.ndarray[float]]:
        """
        Getter for the reduced_costs attribute of the Tab class.
        
        Returns:
        - np.ndarray[float]: The reduced costs of the variables, 1D array.
        
        Raises:
        - ValueError: If the c attribute is not set.
        - ValueError: If the basic_vars attribute is not set.
        - ValueError: If the num_vars attribute is not set.
        """
        if self.c is None:
            raise ValueError("The c attribute is not set. Please set the c attribute before accessing reduced_costs.")

        if self.basic_vars is None:
            raise ValueError("The basic_vars attribute is not set. Please set the basic_vars attribute before accessing reduced_costs.")

        if self._num_vars is None:
            raise ValueError("The num_vars attribute is not set. Please set the num_vars attribute before accessing reduced_costs.")

        if len(self.c) != self._num_vars:
            raise ValueError("The length of the c array does not match the num_vars attribute. Please make sure the length of the c array matches the num_vars attribute before accessing reduced_costs.")

        if len(self.basic_vars) != self.num_constraints:
            raise ValueError("The length of the basic_vars array does not match the num_constraints attribute. Please make sure the length of the basic_vars array matches the num_constraints attribute before accessing reduced_costs.")

        return self._reduced_costs

    @reduced_costs.setter
    def reduced_costs(self, value: np.ndarray[float]) -> None:
        """
        Setter method for the reduced_costs attribute.

        Parameters:
        - value: np.ndarray[float]: The new value for the reduced_costs attribute.

        Raises:
        - ValueError: If the shape of the value array does not match the number of variables in the problem.
        - ValueError: If the phase of the simplex algorithm is 1 and any reduced cost is negative.
        - Warning: If any reduced cost is close to zero (within 1e-10 of zero).

        Suggestions:
        - Check that the value array has the correct shape.
        - Check that the phase of the simplex algorithm is correct.
        - If any reduced cost is negative, the problem may be infeasible or unbounded. Check the constraints and objective function.
        - If any reduced cost is close to zero, the problem may be ill-conditioned. Check the input data and consider using a different solver.

        """
        if value.shape[0] != self._num_vars:
            raise ValueError(f"Invalid shape for reduced costs: expected ({self._num_vars},), got {value.shape}.")
        if self._phase == 1 and np.any(value < 0):
            raise ValueError("Negative reduced costs in phase 1: the problem may be infeasible or unbounded.")
        if np.any(np.isclose(value, 0, atol=1e-10)):
            warnings.warn("Reduced costs are close to zero: the problem may be ill-conditioned.")
        self._reduced_costs = value

    @property
    def basic_vars(self) -> np.ndarray[int]:
        """
        The indices of the variables in the current basis, 1D array.

        Raises:
            ValueError: If the basic_vars attribute has not been set yet.
            ValueError: If the basic_vars attribute has an incorrect shape.
            ValueError: If the basic_vars attribute contains invalid indices.
        """
        if self._basic_vars is None:
            raise ValueError("basic_vars attribute has not been set yet")

        if self._basic_vars.ndim != 1 or self._basic_vars.size != self._num_constraints:
            raise ValueError(f"basic_vars has an incorrect shape. Expected {(self._num_constraints,)}, "
                            f"got {self._basic_vars.shape}")

        if np.min(self._basic_vars) < 0 or np.max(self._basic_vars) >= self._num_vars + self._num_constraints:
            raise ValueError(f"basic_vars contains invalid indices. Expected indices between 0 and {self._num_vars + self._num_constraints - 1}")

        return self._basic_vars

    
    @basic_vars.setter
    def basic_vars(self, value: np.ndarray[int]) -> None:
        """Setter updates the basic_vars attribute and performs error checking.

        Parameters:
        - value: np.ndarray[int]: The indices of the variables in the new basis.

        Raises:
        - ValueError: If value is not a 1D numpy array.
        - ValueError: If the length of value is not equal to num_constraints.
        - ValueError: If any of the indices in value are outside the range [0, num_vars-1].
        - ValueError: If any of the indices in value are duplicated.
        """
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ValueError("basic_vars must be a 1D numpy array")

        if len(value) != self.num_constraints:
            raise ValueError(f"basic_vars length must be equal to num_constraints ({self.num_constraints})")

        if np.any(value < 0) or np.any(value >= self._num_vars):
            raise ValueError(f"basic_vars indices must be between 0 and num_vars-1 ({self._num_vars-1})")

        if len(set(value)) != len(value):
            raise ValueError("basic_vars indices must not contain duplicates")

        self._basic_vars = value

    @property
    def basic_var_values(self) -> np.ndarray[float]:
        """
        The values of the basic variables in the current solution, 1D array.
        Raises a ValueError if the basic_vars attribute has not been set yet.
        Raises a ValueError if the shape of the basic_vars attribute is not compatible with the tableau shape.
        Raises a ValueError if the shape of the basic_var_values attribute is not compatible with the tableau shape.
        """
        if self._basic_vars is None:
            raise ValueError("basic_vars attribute has not been set yet. Please set it first.")
        if len(self._basic_vars) != self._num_constraints:
            raise ValueError(f"Length of basic_vars ({len(self._basic_vars)}) does not match number of constraints ({self._num_constraints}).")
        if self._basic_var_values is None:
            raise ValueError("basic_var_values attribute has not been set yet. Please set it first.")
        if self._basic_var_values.shape != (self._num_constraints,):
            raise ValueError(f"Shape of basic_var_values ({self._basic_var_values.shape}) does not match tableau shape ({self._num_constraints},).")
        return self._basic_var_values

    @basic_var_values.setter
    def basic_var_values(self, value: np.ndarray[float]) -> None:
        """
        Sets the values of the basic variables in the current solution, and calls update_artificial_var_values().

        Parameters:
        value (np.ndarray[float]): The new values of the basic variables.

        Raises:
        TypeError: If value is not a numpy array.
        ValueError: If value does not have the same length as basic_vars, or does not satisfy the constraints.
        Warning: If any values in value are negative.

        Notes:
        - This setter method updates the basic_var_values attribute with the input value, and then calls
          the update_artificial_var_values() method to update the artificial_var_values attribute.
        - The value parameter must be a numpy array with the same length as the basic_vars attribute.
        - The values in value must satisfy the constraints of the linear programming problem.
          If not, a ValueError is raised with a message indicating which constraints are not satisfied.
        - If any values in value are negative, a warning is issued to indicate that the solution may not be feasible.
          It is up to the user to determine if the solution is valid or if additional constraints are required.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("basic_var_values must be a numpy array")
        if len(value) != len(self.basic_vars):
            raise ValueError("basic_var_values must have the same length as basic_vars")
        if not all(val >= 0 for val in value):
            warnings.warn("basic_var_values contains negative values, which may not be feasible")
        if not np.allclose(np.dot(self.A[:, self.basic_vars], value), self.b):
            raise ValueError("basic_var_values does not satisfy the constraints")
                # Check that the basic variable values satisfy the problem constraints
        self._basic_var_values = value
        self.update_artificial_var_values()


    @property
    def slack_vars(self) -> np.ndarray[float]:
        """
        The values of the slack variables in the current solution, 1D array.
        """
        return self._slack_vars


    @slack_vars.setter
    def slack_vars(self, value: np.ndarray[float]) -> None:
        """Sets the values of the slack variables in the current solution.

        Args:
            value (np.ndarray[float]): The values of the slack variables, 1D array.

        Raises:
            ValueError: If the slack vars array does not have the expected size.
            ValueError: If the slack vars are negative.
            ValueError: If the slack vars are not zero for non-binding constraints.

        Notes:
            The slack vars array should have the same size as the number of constraints in the linear programming problem.
            Negative slack vars indicate that the constraint is violated. Non-zero slack vars for non-binding constraints
            indicate that the problem is infeasible. To remedy this, you can try adding a new constraint or modifying
            an existing one to make the problem feasible.
        """
        expected_size = self.num_constraints
        if value.shape != (expected_size,):
            raise ValueError(f"Expected slack vars array of size {expected_size}, but got array of size {value.shape}.")

        if np.any(value < 0):
            raise ValueError("Slack vars cannot be negative.")

        # Check if slack vars are zero for non-binding constraints
        binding_constraints = self._A.dot(self._basic_vars)
        non_binding_constraints = np.setdiff1d(np.arange(self.num_constraints), binding_constraints)
        if np.any(value[non_binding_constraints] != 0):
            raise ValueError("Slack vars should be zero for non-binding constraints.")

        self._slack_vars = value


    @property
    def artificial_var_values(self) -> np.ndarray[float]:
        """
        The values of the artificial variables in the current solution, 1D array.
        
        Raises:
            ValueError: If the artificial variables are not in the basis.
            UserWarning: If any artificial variable has a negative value.
        """
        if self._artificial_var_values is None:
            self.update_artificial_var_values()
        
        # Check that artificial variables are in the basis
        if np.any(np.isin(self.basic_vars, self._artificial_vars)):
            raise ValueError("Artificial variables should not be in the basis.")
        
        # Check for negative values
        if np.any(self._artificial_var_values < 0):
            warnings.warn("Some artificial variables have negative values in the current solution. "
                          "Consider using a different initial feasible solution.",
                          category=UserWarning)
        
        return self._artificial_var_values

    @artificial_var_values.setter
    def artificial_var_values(self, value: np.ndarray[float]) -> None:
        """
        Setter for artificial_var_values attribute.
        
        Args:
        - value (np.ndarray[float]): The new values of the artificial variables.
        
        Raises:
        - ValueError: If value is not a 1D array of floats with the same length as artificial_vars.
        - Warning: If the sum of the absolute values of the artificial_var_values exceeds a small tolerance value.
        
        Suggestions:
        - Check that value is a 1D array of floats with the same length as artificial_vars.
        - Make sure that the sum of the absolute values of the artificial_var_values is small enough.
        """
        if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != np.float:
            raise ValueError("artificial_var_values must be a 1D array of floats")
        
        if value.size != len(self._artificial_vars):
            raise ValueError("artificial_var_values must have the same length as artificial_vars")
        
        if np.sum(np.abs(value)) > 1e-9:
            # warn if the sum of absolute values of artificial_var_values is larger than a small tolerance value
            warnings.warn("The sum of the absolute values of the artificial variables is relatively large. "
                          "Consider re-solving the problem with a larger tolerance value.")
        
        self._artificial_var_values = value


    def update_reduced_costs(self) -> None:
        """
        Updates the reduced_costs attribute using the current tableau and basis.

        This method computes the reduced costs of the variables in the current tableau using the current basis and
        stores the results in the reduced_costs attribute.

        Raises:
            ValueError: If the number of variables or constraints in the tableau is not compatible with the length
                of the coefficients arrays.
            ValueError: If the basic_vars attribute is not set.
            ValueError: If the basic_vars attribute is not a 1D numpy array of integer indices.
            ValueError: If the basic_var_values attribute is not set.
            ValueError: If the basic_var_values attribute is not a 1D numpy array of floats.
            ValueError: If the tableau is degenerate and has multiple optimal solutions.
        """
        if self._c is None or self._A is None or self._b is None:
            raise ValueError("The tableau coefficients are not set.")

        if self._num_vars is None or self._num_constraints is None:
            raise ValueError("The tableau dimensions are not set.")

        if self._num_vars != self._c.shape[0] or self._num_constraints != self._A.shape[0]:
            raise ValueError("The dimensions of the coefficients arrays do not match the tableau dimensions.")

        if self._basic_vars is None:
            raise ValueError("The basic_vars attribute is not set.")

        if not isinstance(self._basic_vars, np.ndarray) or self._basic_vars.ndim != 1 or not np.issubdtype(self._basic_vars.dtype, np.integer):
            raise ValueError("The basic_vars attribute is not a 1D numpy array of integer indices.")

        if self._basic_var_values is None:
            raise ValueError("The basic_var_values attribute is not set.")

        if not isinstance(self._basic_var_values, np.ndarray) or self._basic_var_values.ndim != 1 or not np.issubdtype(self._basic_var_values.dtype, np.floating):
            raise ValueError("The basic_var_values attribute is not a 1D numpy array of floats.")

        # Compute the reduced costs of the non-basic variables
        reduced_costs = np.zeros(self._num_vars)
        for j in range(self._num_vars):
            if j not in self._basic_vars:
                reduced_costs[j] = self._c[j] - np.sum(self._A[:, j] * self._c_full[self._basic_vars])

        self._reduced_costs = reduced_costs

    def update_basic_vars(self) -> None:
        """
        Updates the basic_vars attribute using the current tableau and artificial_vars.

        Raises:
            ValueError: If there is no feasible solution (i.e., all artificial variables are nonzero in the optimal
                        solution of the auxiliary problem).
            ValueError: If the current basis is degenerate (i.e., there are more than n linearly dependent basic
                        variables, where n is the number of variables).
            ValueError: If the current basis is infeasible (i.e., there is no feasible solution to the problem).
        """
        n, m = self.A.shape
        num_artificial_vars = len(self._artificial_vars)
        num_basic_vars = m - num_artificial_vars
        B = self.A[:, self.basic_vars]
        if np.linalg.det(B) == 0:
            raise ValueError("Current basis is degenerate. Try adding more constraints or removing redundant ones.")
        if self._phase == 1 and self.artificial_var_values.sum() > 0:
            raise ValueError("No feasible solution found for the auxiliary problem. Try adding more constraints.")
        if self._phase == 2 and np.any(self.artificial_var_values > 0):
            raise ValueError("Current basis is infeasible. Try adding more constraints or removing redundant ones.")
        if num_basic_vars > n:
            raise ValueError("Current basis is degenerate. Try adding more constraints or removing redundant ones.")
        self._basic_vars = np.zeros(m, dtype=int)
        for i, j in enumerate(self.basic_vars):
            self._basic_vars[j] = i

    def update_basic_var_values(self) -> None:
        """
        Update the values of the basic variables in the current solution
        by solving the system of equations formed by the basic columns of A and b.
        """
        if self._basic_vars is None or self._A is None or self._b is None:
            return

        print("A:",self._A)
        print("basic_vars:",self._basic_vars)
        basic_columns = self._A[:, self._basic_vars[0]].reshape(-1, 1)
        for idx in self._basic_vars[1:]:
            basic_columns = np.column_stack((basic_columns, self._A[:, idx]))

        basic_var_values = np.zeros_like(self._b)

        for i, row in enumerate(basic_columns):
            basic_var_values[i] = self._b[i] / row[i]

        self._basic_var_values = basic_var_values


    def update_artificial_var_values(self) -> None:
        """
        Updates the artificial_var_values attribute using the current tableau and basic_var_values.

        Raises:
        -------
        ValueError:
            If the phase is not 1 or 2.
            If the artificial variables are not in the basis.
            If there are negative artificial variable values in phase 1.
        """
        if self._phase != 1 and self._phase != 2:
            raise ValueError("Invalid phase: phase must be 1 or 2.")
        if not np.all(np.isin(self._artificial_vars, self.basic_vars)):
            raise ValueError("Artificial variables must be in the current basis.")
        if self._phase == 1 and np.any(self.artificial_var_values < 0):
            raise ValueError("Negative artificial variable values in phase 1: "
                            "this indicates an infeasible problem. Try adding more constraints.")

        # Update artificial variable values
        self._artificial_var_values = np.zeros_like(self.basic_var_values)
        self._artificial_var_values[self._artificial_vars == self.basic_vars] = self.basic_var_values[self._artificial_vars == self.basic_vars]
        self._artificial_var_values[self._artificial_vars != self.basic_vars] = self.A[:, self._artificial_vars][self._artificial_vars != self.basic_vars].dot(self.basic_var_values)

    def update_tableau(self) -> None:
        """
        Recomputes the entire tableau from scratch using c, A, and b.
        
        Raises:
            ValueError: If the dimensions of c, A, or b are inconsistent with num_vars or num_constraints, or if any of
                these arrays contains NaN or infinite values.
        """
        # Check for NaN or infinite values in c, A, and b
        if np.isnan(self.c).any() or np.isinf(self.c).any():
            raise ValueError("The objective function coefficients c contain NaN or infinite values.")
        if np.isnan(self.A).any() or np.isinf(self.A).any():
            raise ValueError("The constraint coefficients A contain NaN or infinite values.")
        if np.isnan(self.b).any() or np.isinf(self.b).any():
            raise ValueError("The right-hand side coefficients b contain NaN or infinite values.")

        # Check dimensions of c, A, and b
        if self.c.shape[0] != self._num_vars:
            raise ValueError(f"The number of elements in c ({self.c.shape[0]}) does not match num_vars ({self._num_vars}).")
        if self.A.shape[0] != self.num_constraints or self.A.shape[1] != self._num_vars:
            raise ValueError(f"The dimensions of A ({self.A.shape}) are inconsistent with num_constraints ({self.num_constraints}) and num_vars ({self._num_vars}).")
        if self.b.shape[0] != self.num_constraints:
            raise ValueError(f"The number of elements in b ({self.b.shape[0]}) does not match num_constraints ({self.num_constraints}).")

        # Check for negative or zero values in b
        if (self.b <= 0).any():
            raise ValueError("The right-hand side coefficients b cannot contain negative or zero values. Try adding slack variables.")

        # Check for negative values in A
        if (self.A < 0).any():
            warnings.warn("The constraint coefficients A contain negative values. This may cause unexpected behavior.")

        # Check for a feasible basic solution
        if (self.b < 0).any():
            warnings.warn("The current right-hand side coefficients b do not yield a feasible basic solution. Try adding slack variables or an artificial variable.")

        # Initialize tableau with zeros
        tableau = np.zeros((self.num_constraints + 1, self._num_vars + self.num_constraints + 1))

        # Fill in objective function row
        tableau[0, :self._num_vars] = self.c
        tableau[0, -1] = np.sum(self.c)

        # Fill in constraint rows
        for i in range(self.num_constraints):
            tableau[i+1, :self._num_vars] = self.A[i]
            tableau[i+1, self._num_vars+i] = 1
            tableau[i+1, -1] = self.b[i]

        # Compute slack variables
        self._slack_vars = self.b - np.dot(self.A, tableau[1:, :self._num_vars])

        # Set basic variables and basic variable values
        self._basic_vars = np.arange(self._num_vars, self._num_vars+self.num_constraints)
        self._basic_var_values = self.b

        # Set artificial variables and values for phase 1
        if self._phase == 1:
            self._artificial_vars = []
            for i in range(self.num_constraints):
                if self.b[i] != 0:
                    tableau[i+1, self._num_vars+self.num_constraints+i] = 1
                    self._artificial_vars.append(self._num_vars+self.num_constraints+i)
                    self._artificial_var_values = self.b.copy()

        # Set artificial variables and values for phase 2
        if self._phase == 2:
            if len(self._artificial_vars) > 0:
                # Compute artificial variable values from basic variables
                self._artificial_var_values = np.dot(tableau[1:, self._num_vars+self.num_constraints:], tableau[1:, :self._num_vars])

                # Check if any artificial variable is still in the basis
                if (self.artificial_var_values > 0).any():
                    warnings.warn("The current basic solution still includes one or more artificial variables. Try adding more constraints to eliminate them or use a different initialization method.")
            else:
                self._artificial_var_values = np.zeros(self.num_constraints)

        # Set reduced costs
        self.update_reduced_costs()

        # Update the tableau attribute
        self._tableau = tableau




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

    def loadFromFile(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse objective type and coefficients
        objective_line = lines[0].strip().split(';')
        objective_type = objective_line[0].lower()
        c = np.array([float(coef) for coef in objective_line[1:-1]])
        if objective_type == 'min':
            c = -c
        
        # Parse constraints
        A = []
        b = []
        for constraint_line in lines[1:]:
            constraint_parts = constraint_line.strip().split(';')
            constraint_coeffs = [float(coef) for coef in constraint_parts[:-3]]
            A.append(constraint_coeffs)
            b.append(float(constraint_parts[-2]))
        
        A = np.array(A)
        b = np.array(b)

        # Set the attributes
        self.tab = Tab(c, A, b)


    def solve(self) -> str:
        if self.tab is None:
            raise ValueError("Linear programming problem not defined. Please provide c, A, and b, or load from a file.")

        self._initialize_simplex()

        if self.tab.phase == 1:
            self._phase_one()

        if self.tab.phase == 2:
            self._phase_two()

        return self.status

    def _initialize_simplex(self) -> None:
        """
        Initializes the tableau and sets up phase 1 of the algorithm.
        """

        # Find the indices of the negative b values
        negative_b_indices = np.where(self.tab.b < 0)[0]

        if len(negative_b_indices) > 0:
            # We need to perform Phase 1 to find a feasible solution
            self.tab.phase = 1
            self._setup_phase_one(negative_b_indices)
        else:
            # The initial solution is feasible, so we can skip to Phase 2
            self.tab.phase = 2
            self.tab.basic_vars = np.arange(self.tab.num_vars, self.tab.num_vars + self.tab.num_constraints)
            self.tab.update_reduced_costs()
            self.tab.update_basic_var_values()


    def _setup_phase_one(self, negative_b_indices: np.ndarray[int]) -> None:
        """
        Sets up the auxiliary problem for Phase 1 of the simplex algorithm.
        """
        # Create the auxiliary objective function with artificial variables
        aux_c = np.zeros(self.tab.num_vars + self.tab.num_constraints)
        aux_c[self.tab.num_vars:] = 1

        # Add artificial variables to the constraint matrix
        aux_A = np.hstack((self.tab.A, np.eye(self.tab.num_constraints)))

        # Set artificial variables as the initial basic variables
        aux_basic_vars = np.arange(self.tab.num_vars, self.tab.num_vars + self.tab.num_constraints)

        # Update the Tab instance for the auxiliary problem
        self.tab.c = aux_c
        self.tab.A = aux_A
        self.tab.basic_vars = aux_basic_vars

        # Update the reduced costs, basic variable values, and artificial variable values
        self.tab.update_reduced_costs()
        self.tab.update_basic_var_values()
        self.tab.artificial_var_values = self.tab.basic_var_values.copy()

    def _phase_one(self) -> None:
        """
        Executes Phase 1 of the simplex algorithm.
        """
        iteration = 0

        while not self._is_optimal() and iteration < self.max_iter:
            entering_var = self._choose_entering_variable()
            if entering_var is None:
                break

            leaving_var = self._choose_leaving_variable(entering_var)
            if leaving_var is None:
                self.status = 'unbounded'
                return

            self._pivot(leaving_var, entering_var)

            iteration += 1

        if np.any(self.tab.artificial_var_values > 0):
            self.status = 'infeasible'
            return

        # Remove artificial variables and restore original objective function
        self.tab.A = self.tab.A[:, :self.tab.num_vars]
        self.tab.c = self.tab.c[:self.tab.num_vars]
        self.tab.update_reduced_costs()
        self.tab.update_basic_var_values()

        self.tab.phase = 2



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
simplex.loadFromFile('lp_glaces.txt')

