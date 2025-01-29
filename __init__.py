"""
Time Series Analysis nodes for ComfyUI
"""

from .timeseries_loader import TimeSeriesModelLoader
from .timeseries_predictor import TimeSeriesPredictor
from .Domain_Time_Series_Prep import DomainTimeSeriesPrep

NODE_CLASS_MAPPINGS = {
    "TimeSeriesLoader": TimeSeriesModelLoader,
    "TimeSeriesPredictor": TimeSeriesPredictor,
    "DomainTimeSeriesPrep": DomainTimeSeriesPrep
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimeSeriesLoader": "Load Time Series Model",
    "TimeSeriesPredictor": "Time Series Prediction",
    "DomainTimeSeriesPrep": "Domain Time Series Prep"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
