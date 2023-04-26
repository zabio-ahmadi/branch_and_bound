"""
Authors: 
- Ahmadi (Ahmadi's email)
- Kurteshi (Kurteshi's email)
- Stefan Antun (stefan.antun@hes-so.ch, stefan@logicore.ch)
- Ivan (Ivan's email)

Date: April 21, 2023

Description: This script allows users to solve linear programming (LP) problems using either the Simplex method or 
the Branch and Bound algorithm for Mixed Integer Linear Programming (MILP) problems. The user can provide a problem 
file as a command-line argument and choose which method to use for solving the problem. If no file is provided, the 
default file "examples/lp_glaces.txt" will be used. If no method is provided, the default method is Branch and Bound.

Requirements: Python 3.10 or higher.

Usage:
    python main.py [filename] [--method METHOD]
    where:
    - [filename] (optional): Path to the file containing the linear programming problem. Default is "examples/lp_glaces.txt".
    - METHOD (optional): Method for solving the problem: 'simplex' or 'branchandbound'. Default is 'branchandbound'.
"""

import sys, argparse, os

if sys.version_info < (3, 10):
    sys.exit("Error: This script requires Python 3.10 or higher.")

from src.simplexe import Simplexe
from src.BnB import BranchAndBound


def main(filename: str, method: str):
    if method.lower() == "simplex":
        solver = Simplexe()
        solver.LoadFromFile(filename)
    elif method.lower() == "branchandbound":

        solver = BranchAndBound()
        solver.debug(True)
        solver.go(filename)
    else:
        print("Invalid method provided. Please choose either 'simplex' or 'branchandbound'.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve linear programming problems using Simplex or Branch and Bound methods.")
    parser.add_argument("filename", nargs="?", default="examples/lp_glaces.txt", help="Path to the file containing the linear programming problem. Default is 'examples/lp_glaces.txt'.")
    parser.add_argument("--method", default="branchandbound", help="Method for solving the problem: 'simplex' or 'branchandbound'. Default is 'branchandbound'.")
    args = parser.parse_args()

    if os.path.isfile(args.filename):
        main(args.filename, args.method)
    else:
        print(f"Error: File '{args.filename}' not found. Please provide a valid file path.")