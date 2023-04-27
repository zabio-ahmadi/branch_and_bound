"""
Authors: 
- Ahmadi (Ahmadi's email)
- Kurteshi (Kurteshi's email)
- Stefan Antun (stefan.antun@hes-so.ch, stefan@logicore.ch)
- Ivan (Ivan's email)

Date: April 21, 2023

Description: This script implements the Branch and Bound algorithm to solve Mixed Integer Linear Programming (MILP) problems.
"""

import time, warnings, numpy as np
from copy import deepcopy
from src.simplexe import *
import random


temp_array = []

class BranchAndBound(Simplexe):
    """
    The BranchAndBound class is an extension of the Simplexe class, providing functionality for solving 
    Mixed Integer Linear Programming (MILP) problems using the Branch and Bound algorithm.

    Attributes:
        - index (List): A list containing the indices of the basic variables.
        - depth (int): The depth of the current node in the branch and bound tree.

    Methods:
        - go(lpFileName: str, printDetails: bool = False) -> None:
            Loads a linear programming problem from a file and solves it using the Branch and Bound method.

        - create_bounds(node: 'BranchAndBound', isLeft: bool = True) -> None:
            Creates bounds for the left or right child nodes in the branch and bound tree.

        - PLNE() -> None:
            Implements the Branch and Bound algorithm to solve the Mixed Integer Linear Programming problem.

        - round_numpy_array(arr: np.ndarray, decimals: int = 6) -> np.ndarray:
            Rounds the elements of a numpy array to a specified number of decimal places.

        - pivot(tab: np.ndarray, row: int, col: int) -> None:
            Performs a pivot operation on the given tableau.

        - solve_tableau(tableau: 'BranchAndBound') -> 'BranchAndBound':
            Solves a given tableau using the Simplex method.

        - find_pivot(tableau: np.ndarray) -> tuple[int, int]:
            Finds the pivot element in the given tableau.
    """


    # ------------------------------------------------------------------------------------ __init__
    def __init__(self):
        """
        Initializes a new instance of the BranchAndBound class, inheriting attributes from the Simplexe class.
        """
        super().__init__()
        self.index: List = []
        self.depth: int = 0
        self.DEBUG = False
        self.start_time = time.time()
        self.list_sol = []

    def debug(self,mode = True):
        self.DEBUG = mode


    # ------------------------------------------------------------------------------------ go

    def print_tableau(self, tableau):
        tableau = np.where(np.abs(tableau) < 1e-10, 0, tableau) # Replace -0 and 0 with 0
        np.set_printoptions(suppress=True, linewidth=150)
        np.savetxt(sys.stdout, tableau, fmt=f"%8.3f")
        print()


    def go(self, lpFileName: str, printDetails: bool = False) -> None:
        """
        Load the linear programming problem from the given file and solve it using the simplex method.
        Afterward, round the tableau, print it, and solve the problem using the Branch and Bound method.

        Args:
            - lpFileName (str): The name of the file containing the linear programming problem.
            - printDetails (bool, optional): Whether to print details while solving the problem. Defaults to False.
        """

        self._Simplexe__PrintDetails = printDetails
        self.LoadFromFile(lpFileName, printDetails) # loads file and solves it with the simplex method
        self = self.round_numpy_array(self)
        self.PrintTableau("before PLNE")
        print("------------- start PLNE")
        self.PLNE()

        print("integer solutions:", [float("{:.2f}".format(x)) for x in self.list_sol])
        print("execution time: {:.3f} sec".format(time.time() - self.start_time))
        print("\n------------- finish PLNE")


    # ------------------------------------------------------------------------------------ create_bounds
    def create_bounds(self, node: 'BranchAndBound', isLeft: bool = True) -> None:
        """
        Create and update the tableau with bounds based on the given branching node.
        
        Description:
            - This method creates a new constraint for the tableau based on the given node, depth and isLeft flag.
            The branching is done based on the node's depth and index. The new constraint is appended to the tableau,
            and the tableau is updated accordingly.

        Arguments:
            - node (BranchAndBound): A branching node of the branch and bound algorithm.
            - isLeft (bool, optional): A flag indicating whether to create a left (True) or right (False) bound. Defaults to True.

        Returns:
            - None

        Steps:
            - 1. Find the row and variable that will be used to create the new constraint.
            - 2. Create a new constraint line based on the found row, variable, and isLeft flag.
            - 3. Update the tableau by adding the new constraint.
            - 4. Adjust the basic variables and other attributes of the node accordingly.

        Notes:
            - The input node is modified in-place.
            - The method does not perform any error checking for the input node or the resulting tableau.
            - The method assumes that the input node's tableau and other attributes are valid and up-to-date.

        Raises:
            - ValueError: If the input node's tableau is not a 2D numpy array or has an incorrect shape.
            - ValueError: If he node's depth or index attributes have incorrect types or values.
        """

        # Retrieve the tableau from the input node
        tab: np.ndarray = node._Simplexe__tableau

        # Error checking for input node's tableau, depth, and index
        if not isinstance(tab, np.ndarray) or len(tab.shape) != 2:
            raise ValueError("The input node's tableau must be a 2D numpy array.")
        if not isinstance(node.depth, int) or node.depth < 0:
            raise ValueError("The input node's depth must be a non-negative integer.")
        if not isinstance(node.index, list) or any(not isinstance(i, int) for i in node.index):
            raise ValueError("The input node's index must be a list of integers.")

        # Determine the row and variable for creating the new constraint
        if len(node.index) > 0 : 
            whichRow = node.index[node.depth % len(node.index)]
            whichVariable = node.depth
            whichVariable = idx = np.where(tab[whichRow] == 1.0)[0][0]

            # Calculate the right-hand-side value of the new constraint
            rhs_val = tab[whichRow][-1]
            abs_val = np.floor(rhs_val) if isLeft else np.ceil(rhs_val)

            # Create a new constraint line based on the row, variable, and isLeft flag
            new_line = [0] * (tab.shape[1] - 1) + [1] + [abs_val]
            sign = 1.0 if isLeft else -1.0
            new_line[whichVariable] = sign
            new_line[-1] *= sign

            # Update the tableau by adding the new constraint
            tab = np.hstack((tab[:, :-1], np.atleast_2d([0] * tab.shape[0]).T, tab[:, -1:]))
            tab = np.vstack((tab[:-1], new_line, tab[-1:]))

            # Adjust the tableau according to the new constraint
            if isLeft:
                tab[-2] -= tab[whichRow]
            else:
                tab[-2] += tab[whichRow]
            
            # Update the node's attributes accordingly
            node.NumCols += 1
            node.NumRows += 1
            node._Simplexe__basicVariables = np.append(node._Simplexe__basicVariables, np.max(node._Simplexe__basicVariables)+1)
            node._Simplexe__basicVariables = np.where(node._Simplexe__basicVariables >= node.NumCols, node._Simplexe__basicVariables + 1, node._Simplexe__basicVariables)
            node._Simplexe__tableau = tab


    # ------------------------------------------------------------------------------------ PLNE
    def PLNE(self):
        """
        PLNE (Pseudo-Boolean Linear Programming) method for solving mixed-integer linear programming problems
        using the branch and bound algorithm.

        Description:
            - This method finds an optimal solution to the mixed-integer linear programming problem by recursively 
            creating and exploring nodes in a search tree using the branch and bound algorithm. The method starts 
            with a relaxed linear programming problem, and at each node of the search tree, it creates two 
            branches with integer constraints. The search continues until a feasible integer solution is found 
            or the maximum number of iterations is reached.

        Arguments:
            - None

        Returns:
            - None

        Steps:
            - 1. Check if the current node has an integer objective value and no negative right-hand-side values.
            - 2. If the current node has an integer objective value, update the best solution if necessary.
            - 3. If the current node has a non-integer objective value and all right-hand-side values are non-negative,
            create two child nodes with updated bounds and continue the search.
            - 4. Repeat steps 1-3 until all nodes have been explored or the maximum number of iterations is reached.

        Notes:
            - This method should be called after the initial linear programming problem has been solved using the simplex method.
            - The branch and bound algorithm can be computationally expensive for large-scale problems or problems with many integer variables.

        Raises:
            - TypeError: If any of the inputs to the helper functions are not of the correct type.
        """

        def profondeur_arbre_binaire(arbre, i=0):
            if i >= len(arbre) or arbre[i] is None:
                return 0

            gauche = profondeur_arbre_binaire(arbre, 2 * i + 1)
            droit = profondeur_arbre_binaire(arbre, 2 * i + 2)

            return max(gauche, droit) + 1




        # ------------------------------------------------------------------------------------ PLNE.is_almost_integer
        def is_almost_integer(num: float, threshold: float = 0.01) -> bool:
            """ Helper function to determine if a number is close to an integer within a given threshold """
            return (-1 * threshold < abs(num - round(num)) < threshold)


        # ------------------------------------------------------------------------------------ PLNE.update_nodes
        def update_nodes(node: 'BranchAndBound', list_node: List) -> List:
            """
            Update the list of nodes by creating new nodes with modified bounds based on the current node.

            Arguments:
                - params node ('BranchAndBound'): The current node of the BranchAndBound class.
                - list_node (List): A list containing the nodes to be processed.

            Returns:
                - list_node (List): The updated list of nodes with the new nodes added.

            Steps:
                - 1. Create an index list of the current node's basic variables.
                - 2. Deepcopy the current node and create left and right children by modifying the bounds.
                - 3. Increment the depth of left and right children.
                - 4. Add the left and right children to the list of nodes.
                - 5. Clear the index list of the current node.

            Raises:
                - TypeError: If 'node' is not an instance of BranchAndBound class.
                - TypeError: If 'list_node' is not a list.
                - ValueError: If 'node.index' contains elements that are not integers.
            """
            if not isinstance(node, BranchAndBound):
                raise TypeError("node must be an instance of BranchAndBound class")
            if not isinstance(list_node, list):
                raise TypeError("list_node must be a list")

            rhs_column = node._Simplexe__tableau[:-1,-1].copy()
            rhs_only_frac, _ = np.modf(rhs_column)
            non_int_idxs = np.nonzero(rhs_only_frac)[0]
            #node.index = [i for i in range(len(node._Simplexe__basicVariables)) if node._Simplexe__basicVariables[i] < node.NumCols]
            node.index = [i for i in range(len(node._Simplexe__basicVariables)) if i in non_int_idxs and node._Simplexe__basicVariables[i] < node.NumCols]
            
            if not all(isinstance(i, int) for i in node.index):
                raise ValueError("node.index should only contain integers")

            left_tableau, right_tableau = deepcopy(node), deepcopy(node)
            self.create_bounds(left_tableau, True)
            self.create_bounds(right_tableau, False)

            left_tableau.depth += 1
            right_tableau.depth += 1
            list_node.append(left_tableau)
            list_node.append(right_tableau)

            node.index = []
            return list_node


        # Initialize values and data structures for the branch and bound search
        objval_max = self.ObjValue()
        temp_node = deepcopy(self)
        list_node, z_PLNE = [], float('-inf')
        iteration, max_iterations  = 0,1e4
        list_node = update_nodes(temp_node, list_node)
        
        best_tableau = None
        


        # Main loop for the branch and bound search
        nb_node_fr_ignored = 0 
        nb_node_int_ignored = 0 
        while list_node and iteration < max_iterations:
            # todo:read from end from start, random 
            # statistique de nombre de noeud suprimmé 
            # random_index = random.randint(0, len(list_node) - 1)
            # node = list_node.pop(random_element) # remove random 
            # node = list_node.pop(0) # remove from begining 
            node = list_node.pop() # remove from end 
            
            node = self.solve_tableau(node)
            # Get the objective value and print it
            objval = node.ObjValue()

            # Check if the current node has a better objective value than the best found so far
            if objval > objval_max:
                nb_node_fr_ignored += 1
                continue # no need to check anything if current objective value is worse than best
            
 
            # Check if the current node has an integer objective value and no negative right-hand-side values
            isInteger = is_almost_integer(objval)
            iteration += 1

            if isInteger:
                
                self.list_sol.append(objval)
                if self.DEBUG:
                    for element in list_node:  # parcours tous les éléments du tableau d'entrée
                        temp_array.append(element)
                        
                 
                    print("Integer solution found")
                    self.print_tableau(node._Simplexe__tableau)
                    print("OBJECTIVE VALUE : {:.2f} ".format(objval))
                    print("solution Depth", profondeur_arbre_binaire(temp_array), "\n")
          

                # if current node is better than previous best node than change it
                if objval > z_PLNE:
                    z_PLNE = objval
                    best_tableau = node
                else: 
                    nb_node_int_ignored += 1
                    continue



            # Update the search tree nodes if the current node has a feasible non-integer solution
            elif all(t >= 0 for t in node._Simplexe__tableau[:-1, -1]):  # Update nodes only when the current node has a feasible solution
                list_node = update_nodes(node, list_node)

        # Print the best solution found after searching all nodes
        if best_tableau:
            print("\nBest Integer solution")
            self.print_tableau(best_tableau._Simplexe__tableau)
            print("OBJECTIVE VALUE : {:.2f}".format(z_PLNE))
            print("solution Depth", profondeur_arbre_binaire(temp_array))
            print(f"Number of node ignored {nb_node_int_ignored}")
    # ------------------------------------------------------------------------------------ round_numpy_array
    def round_numpy_array(self, arr: 'BranchAndBound', decimals: int = 6) -> np.ndarray:
        """
        Round the elements of a given numpy array up to the specified number of decimal places.

        Args:
            - arr (np.ndarray): The input numpy array to be rounded.
            - decimals (int, optional): The number of decimal places to round to. Defaults to 6.

        Returns:
            - np.ndarray: The rounded numpy array.

        Raises:
            - ValueError: If the input is not a numpy array or if decimals is not a non-negative integer.
            - TypeError: If the elements of the input numpy array are not numeric.

        Notes:
            - This function rounds the input numpy array in place.
            - Non-numeric elements in the input array are not modified.
        """

        #if not isinstance(arr, np.ndarray):
        #    raise ValueError("Input 'arr' must be a numpy array.")
        #if not isinstance(decimals, int) or decimals < 0:
        #    raise ValueError("Input 'decimals' must be a non-negative integer.")

        try:
            # convert the array to a floating-point type
            arr._Simplexe__tableau = arr._Simplexe__tableau.astype(float)

            # round the array to a maximum of 6 decimal places
            arr._Simplexe__tableau = np.round(arr._Simplexe__tableau, decimals=decimals)

        except (ValueError, TypeError) as e:
            raise TypeError("All elements of the input numpy array must be numeric.") from e

        # convert any non-float values back to strings
        for i in range(arr._Simplexe__tableau.shape[0]):
            for j in range(arr._Simplexe__tableau.shape[1]):
                if not isinstance(arr._Simplexe__tableau[i, j], float):
                    arr._Simplexe__tableau[i, j] = str(arr._Simplexe__tableau[i, j])

        return arr


    # ------------------------------------------------------------------------------------ pivot
    def pivot(self, tab: np.ndarray, row: int, col: int) -> None:
        """
        Perform a pivot operation on the given tableau at the specified row and column.

        This method modifies the tableau in-place and does not return any value.

        Arguments:
            - tab (np.ndarray): The tableau (2D numpy array) to perform the pivot operation on
            - row (int): The row index of the pivot element in the tableau
            - col (int): The column index of the pivot element in the tableau

        Steps:
            - 1. Divide the pivot row by the pivot element to normalize the pivot row.
            - 2. Update all other rows by subtracting the appropriate multiple of the normalized pivot row,
            to eliminate the entries in the pivot column.

        Raises:
            - ValueError: If row or col are out of bounds for the tableau dimensions
            - ZeroDivisionError: If the pivot element is zero or very close to zero

        Notes:
            - The pivot operation is a fundamental operation in the simplex method for linear programming.
            - It is used to iteratively improve the current solution until an optimal solution is found.
        """

        if row < 0 or row >= tab.shape[0] or col < 0 or col >= tab.shape[1]:
            raise ValueError("Row or column index out of bounds for the tableau dimensions.")

        pivot_element = tab[row, col]

        if np.isclose(pivot_element, 0):
            raise ZeroDivisionError("Pivot element is zero or very close to zero. Cannot perform pivot operation.")

        pivot_row: np.ndarray = tab[row, :]
        pivot_row /= pivot_element
        rows_to_update = np.arange(tab.shape[0]) != row
        tab[rows_to_update, :] -= tab[rows_to_update, col, np.newaxis] * pivot_row


    # ------------------------------------------------------------------------------------ solve_tableau
    def solve_tableau(self, tableau: 'BranchAndBound') -> 'BranchAndBound':
        """
        Solves a given tableau using the simplex method.

        Arguments:
            - tableau (BranchAndBound): The BranchAndBound object containing the tableau to be solved.

        Returns:
            - BranchAndBound: The solved BranchAndBound object containing the updated tableau.

        Steps:
            - 1. Find the pivot element.
            - 2. Perform the pivot operation.
            - 3. Continue the process until the tableau is optimized or a warning is raised.

        Notes:
            - The method modifies the given BranchAndBound object in-place.
            - If the tableau is not in the correct format, it raises a ValueError.
            - If no optimal solution is found, it raises a warning.

        Raises:
            - ValueError: If the tableau is not in the correct format (should be a 2D numpy array).
            - Warning: If no optimal solution is found.
        """

        if not isinstance(tableau, BranchAndBound):
            raise ValueError("The input must be a BranchAndBound object.")
        
        tab: np.ndarray = tableau._Simplexe__tableau
        
        if not isinstance(tab, np.ndarray) or len(tab.shape) != 2:
            raise ValueError("The tableau must be a 2D numpy array.")

        iteration, max_iterations = 0, 1000

        while True:
            if iteration >= max_iterations:
                warnings.warn("Maximum number of iterations reached. No optimal solution found.")
                break

            row, col = self.find_pivot(tab)
            if row is None:
                break
            try:
                self.pivot(tab, row, col)
            except:
                break
        tableau = self.round_numpy_array(tableau)
        return tableau


    # ------------------------------------------------------------------------------------ find_pivot
    def find_pivot(self, tableau: np.ndarray) -> tuple[int, int]:
        """
        Find the pivot row and column in the given tableau for the Branch and Bound method.

        Description:
            - This method finds the first row with a negative right-hand-side value, and then
            chooses the column with the highest ratio of column value to the objective function
            value in that row.

        Arguments:
            - tableau (np.ndarray): A 2D numpy array representing the tableau.

        Returns:
            - tuple[int, int]: A tuple containing the pivot row and column indices. If no
            pivot is found, the function returns (None, None).

        Steps:
            - 1. Find the first row with a negative right-hand-side value.
            - 2. Find the column with the highest ratio of column value to the objective function
            value in the row found in step 1.

        Raises:
            - ValueError: If the input 'tableau' is not a numpy array.
            - ValueError: If the input 'tableau' is not a 2D numpy array.
            - ValueError: If the input 'tableau' has less than two rows or columns.

        Notes:
            - This method is used by the Branch and Bound method for solving mixed integer linear
            programming problems.
            - The tableau must be a 2D numpy array containing at least two rows and two columns.
        """

        # Error checking
        if not isinstance(tableau, np.ndarray):
            raise ValueError("Input 'tableau' must be a numpy array.")
        if tableau.ndim != 2:
            raise ValueError("Input 'tableau' must be a 2D numpy array.")
        if tableau.shape[0] < 2 or tableau.shape[1] < 2:
            raise ValueError("Input 'tableau' must have at least two rows and two columns.")
            
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
