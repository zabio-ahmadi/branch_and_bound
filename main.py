"""
  Author:    Niklaus Eggenberg
  Created:   23.02.2023
  Main function - put file as command line argument
  Example of use: python main.py C:\HEPIA\neg-teaching-material\2022-2023\2e\lp_glaces.txt
"""
from src.BnB import BranchAndBound

filename = "examples/lp_glaces.txt"



if "__main__" == __name__:
  bb = BranchAndBound()
  bb.go(filename)