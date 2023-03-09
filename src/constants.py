"""
  Author:    Niklaus Eggenberg
  Created:   23.02.2023
  Constants container file
"""
from enum import Enum

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