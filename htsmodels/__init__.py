__version__ = "0.3.21"

from htsmodels import models
from htsmodels import results

# Only print in interactive mode
import __main__ as main
if not hasattr(main, '__file__'):
    print("""Importing the htsmodels module. L. Roque. 
    Forecasting models for Hierarchical Time Series Forecasting Algorithms.\n""")