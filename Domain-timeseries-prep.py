import torch
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import yfinance as yf  # for stock data
import meteostat as mt  # for weather data
import folder_paths

class DomainTimeSeriesPrep:
    """
    Enhanced ComfyUI node for preparing domain-specific time series data
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.models_dir = os.path.join(folder_paths.models_dir, "timeseries")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "domain_type": ([
                    "auto_detect",
                    "stock_market",
                    "weather",
                    "sensor_data",
                    "satellite",
                    "traffic",
                    "custom"
                ]),
                "input_path": ("STRING", {
                    "multiline": False,
                    "default": "path/to/data"
                }),
                "model_name": ("STRING", {
                    "multiline": False,
                    "default": "model_folder_name"
                }),
            },
            "optional": {
                "force_format": (["true", "false"], {"default": "true"}),
                "data_frequency": ([
                    "auto",
                    "1min", 
                    "5min", 
                    "15min", 
                    "30min",
                    "1H",
                    "1D"
                ], {"default": "auto"})
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("prepared_data",)
    FUNCTION = "prepare_domain_data"
    CATEGORY = "Time Series"

    def prepare_domain_data(self, domain_type, input_path, model_name, 
                          force_format="true", data_frequency="auto"):
        try:
            # Load model config
            config = self._load_model_config(model_name)
            
            # Auto-detect domain if needed
            if domain_type == "auto_detect":
                domain_type = self._detect_domain(input_path)
                print(f"Detected domain type: {domain_type}")
            
            # Load and format data based on domain
            raw_data = self._load_domain_data(domain_type, input_path, data_frequency)
            
            # Apply domain-specific formatting
            formatted_data = self._format_domain_data(
                raw_data, 
                domain_type, 
                config, 
                force_format
            )
            
            # Convert to tensor
            data_tensor = torch.tensor(formatted_data.values, dtype=torch.float32)
            
            return (data_tensor,)
            
        except Exception as e:
            print(f"Error preparing domain data: {str(e)}")
            raise e

    def _detect_domain(self, input_path):
        """Auto-detect data domain based on content"""
        try:
            sample_data = pd.read_csv(input_path, nrows=5)
            columns = set(sample_data.columns.str.lower())
            
            # Check for domain-specific indicators
            if any(col in columns for col in ['open', 'high', 'low', 'close', 'volume']):
                return "stock_market"
            elif any(col in columns for col in ['temp', 'humidity', 'pressure']):
                return "weather"
            elif any(col in columns for col in ['sensor', 'reading', 'measurement']):
                return "sensor_data"
            elif any(col in columns for col in ['latitude', 'longitude', 'band']):
                return "satellite"
            elif any(col in columns for col in ['speed', 'count', 'vehicles']):
                return "traffic"
            else:
                return "custom"
        except:
            return "custom"

    def _load_domain_data(self, domain_type, path, frequency):
        """Load data with domain-specific handling"""
        if domain_type == "stock_market":
            return self._load_stock_data(path, frequency)
        elif domain_type == "weather":
            return self._load_weather_data(path, frequency)
        elif domain_type == "sensor_data":
            return self._load_sensor_data(path, frequency)
        elif domain_type == "satellite":
            return self._load_satellite_data(path)
        elif domain_type == "traffic":
            return self._load_traffic_data(path, frequency)
        else:  # custom
            return pd.read_csv(path)

    def _format_domain_data(self, data, domain_type, config, force_format):
        """Apply domain-specific formatting"""
        if domain_type == "stock_market":
            return self._format_stock_data(data, config, force_format)
        elif domain_type == "weather":
            return self._format_weather_data(data, config, force_format)
        elif domain_type == "sensor_data":
            return self._format_sensor_data(data, config, force_format)
        elif domain_type == "satellite":
            return self._format_satellite_data(data, config, force_format)
        elif domain_type == "traffic":
            return self._format_traffic_data(data, config, force_format)
        else:
            return self._format_custom_data(data, config, force_format)

    def _format_stock_data(self, data, config, force_format):
        """Format stock market data"""
        # Ensure OHLCV format
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        if force_format == "true":
            # Add missing columns with reasonable defaults
            for col in required_cols:
                if col not in data.columns:
                    if col == 'volume':
                        data[col] = 0
                    else:
                        data[col] = data['close'] if 'close' in data.columns else data.iloc[:,0]
        
        # Calculate additional features
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        return data.fillna(method='ffill')

    def _format_weather_data(self, data, config, force_format):
        """Format weather data"""
        # Handle common weather variables
        weather_vars = ['temperature', 'humidity', 'pressure', 'precipitation']
        
        if force_format == "true":
            # Convert temperature if needed
            if 'temperature_f' in data.columns:
                data['temperature'] = (data['temperature_f'] - 32) * 5/9
            
            # Normalize humidity to 0-100
            if 'humidity' in data.columns:
                data['humidity'] = data['humidity'].clip(0, 100)
        
        # Add derived features
        if 'temperature' in data.columns and 'humidity' in data.columns:
            data['heat_index'] = self._calculate_heat_index(
                data['temperature'], 
                data['humidity']
            )
        
        return data

    def _format_sensor_data(self, data, config, force_format):
        """Format sensor data"""
        # Handle missing values
        data = data.interpolate(method='time')
        
        # Remove outliers
        if force_format == "true":
            for col in data.select_dtypes(include=[np.number]).columns:
                data[col] = self._remove_outliers(data[col])
        
        return data

    def _calculate_heat_index(self, temp, humidity):
        """Calculate heat index from temperature and humidity"""
        return temp + 0.555 * (humidity/100) * (temp - 14.5)

    def _remove_outliers(self, series, n_std=3):
        """Remove outliers based on standard deviation"""
        mean = series.mean()
        std = series.std()
        return series.clip(mean - n_std * std, mean + n_std * std)

# Node registration
NODE_CLASS_MAPPINGS = {
    "DomainTimeSeriesPrep": DomainTimeSeriesPrep
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DomainTimeSeriesPrep": "Domain Time Series Prep"
}
