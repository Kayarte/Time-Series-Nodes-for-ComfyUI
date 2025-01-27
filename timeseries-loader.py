import os
import torch
import folder_paths
from transformers import AutoModel

class TimeSeriesModelLoader:
    """
    ComfyUI node for loading time series models from local directory
    """
    
    def __init__(self):
        self.model_path = os.path.join(folder_paths.models_dir, "timeseries")
        # Create timeseries directory if it doesn't exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
    @classmethod
    def INPUT_TYPES(cls):
        # Get list of available models
        ts_path = os.path.join(folder_paths.models_dir, "timeseries")
        if not os.path.exists(ts_path):
            available_models = []
        else:
            available_models = [f for f in os.listdir(ts_path) 
                              if os.path.isdir(os.path.join(ts_path, f))]
        
        return {
            "required": {
                "model_name": (available_models, ),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }
    
    RETURN_TYPES = ("MODEL", )
    RETURN_NAMES = ("model", )
    FUNCTION = "load_model"
    CATEGORY = "Time Series"

    def load_model(self, model_name, device="cuda"):
        try:
            model_dir = os.path.join(self.model_path, model_name)
            
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory {model_name} not found in {self.model_path}")
            
            # Load model configuration
            config_path = os.path.join(model_dir, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found in {model_dir}")
            
            # Load the model
            model = AutoModel.from_pretrained(model_dir)
            model.to(device)
            
            print(f"Successfully loaded model: {model_name}")
            return (model,)
            
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            raise e

    @classmethod
    def IS_CHANGED(cls, model_name, device="cuda"):
        # Track if model files have changed
        model_path = os.path.join(folder_paths.models_dir, "timeseries", model_name)
        if not os.path.exists(model_path):
            return float("nan")
        return os.path.getmtime(model_path)

# Node registration
NODE_CLASS_MAPPINGS = {
    "TimeSeriesLoader": TimeSeriesModelLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimeSeriesLoader": "Load Time Series Model"
}
