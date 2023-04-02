"""
  Author:    Niklaus Eggenberg
  Created:   23.02.2023
  Main function - put file as command line argument
  Example of use: python main.py C:\HEPIA\neg-teaching-material\2022-2023\2e\lp_glaces.txt
"""
import sys
from src.lp import LP
from src.simplexe import Simplexe

splx = Simplexe()
# set log level to True to print intermediate tableaus
splx.LoadFromFile('examples/lp_glaces.txt', True)
