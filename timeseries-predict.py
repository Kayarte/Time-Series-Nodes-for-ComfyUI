import torch
import numpy as np
import pandas as pd
from PIL import Image
import folder_paths
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
                "input_data": ("TENSOR",),
                "prediction_strategy": ([
                    "single_step",    # One step ahead
                    "multi_step",     # Multiple steps ahead
                    "rolling"         # Rolling prediction
                ], {
                    "default": "single_step"
                }),
                "prediction_length": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "output_format": ([
                    "raw",           # Just the numbers
                    "with_dates",    # Include timestamps
                    "full_analysis"  # Additional metrics
                ], {
                    "default": "raw"
                }),
                "visualization_type": (["line", "heatmap"], {
                    "default": "line"
                })
            },
            "optional": {
                "confidence_interval": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 0.99,
                    "step": 0.01
                }),
                "start_date": ("STRING", {
                    "default": "now",  # For timestamp generation
                    "multiline": False
                })
            }
        }
    
    RETURN_TYPES = ("TENSOR", "IMAGE",)
    RETURN_NAMES = ("prediction", "visualization",)
    FUNCTION = "predict"
    CATEGORY = "Time Series"

    def predict(self, model, input_data, prediction_strategy, prediction_length, 
                output_format, visualization_type="line", confidence_interval=0.95,
                start_date="now"):
        try:
            # Handle prediction strategy
            if prediction_strategy == "single_step":
                predictions = self._single_step_prediction(model, input_data)
            elif prediction_strategy == "multi_step":
                predictions = self._multi_step_prediction(model, input_data, prediction_length)
            else:  # rolling
                predictions = self._rolling_prediction(model, input_data, prediction_length)
            
            # Calculate confidence intervals if model supports it
            try:
                lower_bound, upper_bound = self._calculate_confidence_intervals(
                    predictions, confidence_interval
                )
                has_confidence = True
            except:
                has_confidence = False
                
            # Format output
            formatted_output = self._format_output(
                predictions, 
                output_format, 
                start_date, 
                lower_bound if has_confidence else None,
                upper_bound if has_confidence else None
            )
            
            # Create visualization
            viz_array = self._create_visualization(
                input_data,
                formatted_output,
                visualization_type,
                has_confidence,
                lower_bound if has_confidence else None,
                upper_bound if has_confidence else None
            )
            
            return (formatted_output, viz_array)
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise e
    
    def _single_step_prediction(self, model, input_data):
        """Make a single step prediction"""
        with torch.no_grad():
            return model(input_data)
    
    def _multi_step_prediction(self, model, input_data, steps):
        """Make multiple step prediction"""
        predictions = []
        current_input = input_data
        
        with torch.no_grad():
            for _ in range(steps):
                pred = model(current_input)
                predictions.append(pred)
                # Update input for next prediction
                current_input = self._update_input_tensor(current_input, pred)
        
        return torch.cat(predictions, dim=0)
    
    def _rolling_prediction(self, model, input_data, steps):
        """Make rolling predictions updating with each step"""
        predictions = []
        current_input = input_data
        
        with torch.no_grad():
            for _ in range(steps):
                pred = model(current_input)
                predictions.append(pred)
                # Update input with actual data if available
                current_input = self._update_input_tensor(current_input, pred)
        
        return torch.cat(predictions, dim=0)
    
    def _update_input_tensor(self, input_tensor, new_prediction):
        """Update input tensor with new prediction for next step"""
        # Remove oldest timestep and add new prediction
        updated = torch.cat([input_tensor[:, 1:], new_prediction.unsqueeze(1)], dim=1)
        return updated
    
    def _format_output(self, predictions, output_format, start_date, lower_bound=None, upper_bound=None):
        """Format predictions according to specified output format"""
        if output_format == "raw":
            return predictions
            
        elif output_format == "with_dates":
            # Generate timestamps
            if start_date == "now":
                start = datetime.now()
            else:
                start = datetime.strptime(start_date, "%Y-%m-%d")
            
            dates = [start + timedelta(days=i) for i in range(len(predictions))]
            return {
                'dates': dates,
                'predictions': predictions
            }
            
        else:  # full_analysis
            metrics = self._calculate_metrics(predictions)
            return {
                'predictions': predictions,
                'metrics': metrics,
                'confidence_bounds': {
                    'lower': lower_bound,
                    'upper': upper_bound
                } if lower_bound is not None else None
            }
    
    def _calculate_metrics(self, predictions):
        """Calculate additional metrics for full analysis"""
        return {
            'mean': float(torch.mean(predictions)),
            'std': float(torch.std(predictions)),
            'min': float(torch.min(predictions)),
            'max': float(torch.max(predictions))
        }
    
    def _create_visualization(self, input_data, predictions, viz_type, 
                            has_confidence, lower_bound=None, upper_bound=None):
        """Create visualization based on type"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if viz_type == "line":
            # Plot historical data
            ax.plot(input_data.cpu().numpy(), label='Historical', color='blue')
            
            # Plot predictions
            pred_x = np.arange(len(input_data), len(input_data) + len(predictions))
            ax.plot(pred_x, predictions.cpu().numpy(), label='Prediction', color='red')
            
            if has_confidence:
                ax.fill_between(pred_x, lower_bound, upper_bound, 
                              color='red', alpha=0.2, label='Confidence Interval')
            
        else:  # heatmap
            plt.imshow(predictions.cpu().numpy(), aspect='auto', cmap='viridis')
            plt.colorbar(label='Predicted Value')
        
        ax.set_title('Time Series Prediction')
        ax.legend()
        
        # Save and convert to numpy array
        plt_path = f"{self.output_dir}/prediction_plot.png"
        plt.savefig(plt_path)
        plt.close()
        
        viz_image = Image.open(plt_path)
        return np.array(viz_image)

# Node registration
NODE_CLASS_MAPPINGS = {
    "TimeSeriesPredictor": TimeSeriesPredictor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimeSeriesPredictor": "Time Series Prediction"
}
