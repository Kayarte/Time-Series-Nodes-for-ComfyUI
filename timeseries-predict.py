import torch
import numpy as np
from PIL import Image
import folder_paths
import matplotlib.pyplot as plt

class TimeSeriesPredictor:
    """
    ComfyUI node for running predictions with time series models
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "input_data": ("TENSOR",),  # Time series input data
                "prediction_length": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "visualization_type": (["line", "heatmap"], {
                    "default": "line"
                }),
            },
            "optional": {
                "confidence_interval": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 0.99,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "IMAGE",)
    RETURN_NAMES = ("prediction", "visualization",)
    FUNCTION = "predict"
    CATEGORY = "Time Series"

    def predict(self, model, input_data, prediction_length=1, 
                visualization_type="line", confidence_interval=0.95):
        try:
            # Ensure input is on same device as model
            device = next(model.parameters()).device
            input_data = input_data.to(device)
            
            # Run prediction
            with torch.no_grad():
                predictions = model.generate(
                    input_data,
                    max_length=prediction_length,
                    num_return_sequences=1
                )
            
            # Calculate confidence intervals if model supports it
            try:
                lower_bound, upper_bound = self._calculate_confidence_intervals(
                    model, predictions, confidence_interval
                )
                has_confidence = True
            except:
                has_confidence = False
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if visualization_type == "line":
                # Plot input data
                ax.plot(input_data.cpu().numpy(), label='Input', color='blue')
                # Plot prediction
                pred_x = np.arange(len(input_data), len(input_data) + prediction_length)
                ax.plot(pred_x, predictions.cpu().numpy(), label='Prediction', color='red')
                
                if has_confidence:
                    ax.fill_between(pred_x, lower_bound, upper_bound, 
                                  color='red', alpha=0.2, label='Confidence Interval')
                
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Value')
                ax.legend()
                
            else:  # heatmap
                # Create heatmap of predictions
                plt.imshow(predictions.cpu().numpy().T, aspect='auto', cmap='viridis')
                plt.colorbar(label='Predicted Value')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Features')
            
            # Save and load visualization
            viz_path = f"{self.output_dir}/prediction_viz.png"
            plt.savefig(viz_path)
            plt.close()
            
            viz_image = Image.open(viz_path)
            viz_array = np.array(viz_image)
            
            return (predictions, viz_array)
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise e
    
    def _calculate_confidence_intervals(self, model, predictions, confidence_level):
        """
        Calculate confidence intervals if model supports it
        Returns lower and upper bounds
        """
        try:
            # This is a placeholder - actual implementation would depend on model type
            std = torch.std(predictions, dim=0)
            z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor((1 + confidence_level) / 2))
            
            mean = torch.mean(predictions, dim=0)
            lower = mean - z_score * std
            upper = mean + z_score * std
            
            return lower, upper
        except:
            raise NotImplementedError("Confidence intervals not supported for this model")

# Node registration
NODE_CLASS_MAPPINGS = {
    "TimeSeriesPredictor": TimeSeriesPredictor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimeSeriesPredictor": "Time Series Prediction"
}
