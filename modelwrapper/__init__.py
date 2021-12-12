__version__ = "0.0.1"

from modelwrapper import models
from modelwrapper import results

# Only print in interactive mode
import __main__ as main
if not hasattr(main, '__file__'):
    print("""Importing the modelwrapper module. L. Roque. 
    Forecasting models for Hierarchical Time Series Forecasting Algorithms.\n""")