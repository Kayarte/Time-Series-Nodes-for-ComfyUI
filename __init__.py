"""
Time Series Analysis nodes for ComfyUI
"""

from .timeseries_loader import TimeSeriesModelLoader
from .timeseries_predictor import TimeSeriesPredictor
from .timeseries_prep import TimeSeriesPrep

NODE_CLASS_MAPPINGS = {
    "TimeSeriesLoader": TimeSeriesModelLoader,
    "TimeSeriesPredictor": TimeSeriesPredictor,
    "TimeSeriesPrep": TimeSeriesPrep
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimeSeriesLoader": "Load Time Series Model",
    "TimeSeriesPredictor": "Time Series Prediction",
    "TimeSeriesPrep": "Time Series Data Prep"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
