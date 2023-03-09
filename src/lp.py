"""
  Author:    Niklaus Eggenberg
  Created:   23.02.2023
  LP Class - contains the LP structure definition
"""
import numpy as np
import os.path
from os import path
from src.constants import *

class LP:
  def __init__(self, lpFileName):
    self.FileName = lpFileName
    self.Costs =  np.array([], dtype=float)
    self.AMatrix = np.array([], dtype=float)
    self.RHS = np.array([], dtype=float)
   
  def ParseFile(self):
    print("Parsing ", self.FileName)
    if not path.isfile(self.FileName):
      print("Error: file " + self.FileName + " is not defined")
      return False

    with open(self.FileName) as file:
      iCount = 0
      self.__tempB = []
      self.__tempA = []
      for line in file:
        # first line contains objective function type and coefficients
        if iCount == 0:
          self.parseObjective(line)
        else:
          # remaining lines define constraints
          self.parseConstraint(line)
        iCount += 1
    self.AMatrix = np.array(self.__tempA, dtype=float)
    self.RHS = np.array(self.__tempB, dtype=float)
    return True

  def parseObjective(self, sLine):
    lineValues = sLine.rstrip().strip(";").split(";")
    if 'max' in lineValues[0].strip().lower():
      self.ObjectiveType = OptimizationType.Max
      self.Costs = np.array(lineValues[1:], dtype=float)*-1.
    else:
      self.ObjectiveType = OptimizationType.Min
      self.Costs = np.array(lineValues[1:], dtype=float)
      
  def parseConstraint(self, sLine):
    lineValues = sLine.rstrip().strip(";").split(";")
    #skip emtpy lines!
    if len(lineValues) <= 1:
      return
    coeffs = np.array(lineValues[0:-2], dtype=float)
    # <= adds coefficients as they are
    if "<=" in lineValues[-2]:
      self.__tempA.append(coeffs.tolist())
      self.__tempB.append(lineValues[-1])
    # convert >= to <= by inverting signs
    elif ">=" in lineValues[-2]:
      self.__tempA.append(np.multiply(coeffs, -1.).tolist())
      self.__tempB.append(float(lineValues[-1])*-1.)
    #consider any other as an equality => add both <= and >= constraints, but directly convert >= to <=...
    else:
      # <= contraint
      self.__tempA.append(coeffs.tolist())
      self.__tempB.append(lineValues[-1])
      # >= contraint (converted to <= by multiplying by -1)
      self.__tempA.append(np.multiply(coeffs, -1.).tolist())
      self.__tempB.append(float(lineValues[-1])*-1.)

  def PrintProblem(self):
    print("Costs: ", self.Costs)
    print("AMatrix: ", self.AMatrix)
    print("RHS: ", self.RHS)


      