#!/usr/bin/env python3
"""creates a pd.DataFrame from a dictionary"""
import pandas as pd
import numpy as np


df = pd.DataFrame({'First': pd.Categorical([0.0, 0.5, 1.0, 1.5]),
                   'Second': pd.Categorical(["one", "two", "three", "four"])},
                  index=pd.Categorical(['A', 'B', 'C', 'D']))
