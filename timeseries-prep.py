import torch
import numpy as np
import pandas as pd
from PIL import Image
import folder_paths

class TimeSeriesPrep:
    """
    ComfyUI node for preparing time series data
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_type": (["csv", "numpy", "image_sequence"],),
                "sequence_length": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "input_path": ("STRING", {
                    "multiline": False,
                    "default": "path/to/data"
                }),
            },
            "optional": {
                "normalize": (["true", "false"], {
                    "default": "true"
                }),
                "target_column": ("STRING", {
                    "multiline": False,
                    "default": "value"
                }),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("prepared_data",)
    FUNCTION = "prepare_data"
    CATEGORY = "Time Series"

    def prepare_data(self, input_type, sequence_length, input_path, 
                    normalize="true", target_column="value"):
        try:
            # Load data based on input type
            if input_type == "csv":
                data = self._load_csv(input_path, target_column)
            elif input_type == "numpy":
                data = self._load_numpy(input_path)
            else:  # image_sequence
                data = self._load_image_sequence(input_path)
            
            # Convert to tensor
            data_tensor = torch.tensor(data, dtype=torch.float32)
            
            # Normalize if requested
            if normalize == "true":
                data_tensor = self._normalize_data(data_tensor)
            
            # Create sequences
            sequences = self._create_sequences(data_tensor, sequence_length)
            
            return (sequences,)
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            raise e
    
    def _load_csv(self, path, target_column):
        """Load and process CSV data"""
        df = pd.read_csv(path)
        return df[target_column].values
    
    def _load_numpy(self, path):
        """Load numpy array"""
        return np.load(path)
    
    def _load_image_sequence(self, path):
        """Load sequence of images"""
        # Implementation depends on how images are stored
        pass
    
    def _normalize_data(self, tensor):
        """Normalize the data to [0,1] range"""
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        return (tensor - min_val) / (max_val - min_val)
    
    def _create_sequences(self, data, sequence_length):
        """Create sequences for time series prediction"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequence = data[i:(i + sequence_length)]
            sequences.append(sequence)
        return torch.stack(sequences)

# Node registration
NODE_CLASS_MAPPINGS = {
    "TimeSeriesPrep": TimeSeriesPrep
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimeSeriesPrep": "Time Series Data Prep"
}
