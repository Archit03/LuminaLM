from typing import Dict, Any, Optional, Union
import torch
from PIL import Image
import logging
from pathlib import Path
from transformers import ViTImageProcessor, CLIPProcessor
import numpy as np
from torchvision import transforms
from dataclasses import dataclass

@dataclass
class ImageProcessingConfig:
    image_size: int = 224
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    max_pixel_value: int = 255
    to_rgb: bool = True

class ImageFeatureExtractor:
    """Enhanced image feature extraction with multiple model support"""
    
    def __init__(
        self,
        model_type: str = 'ViT',
        model_name: str = 'google/vit-base-patch16-224',
        device: Optional[str] = None,
        config: Optional[ImageProcessingConfig] = None
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or ImageProcessingConfig()
        
        # Initialize processors based on model type
        try:
            if model_type == 'ViT':
                self.processor = ViTImageProcessor.from_pretrained(model_name)
            elif model_type == 'CLIP':
                self.processor = CLIPProcessor.from_pretrained(model_name)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Set up basic transforms
            self.transforms = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.mean, std=self.config.std)
            ])
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize image processor: {str(e)}")

    def preprocess_image(
        self,
        image: Union[Image.Image, str, Path],
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess image with enhanced error handling and validation.
        """
        try:
            # Load image if path is provided
            if isinstance(image, (str, Path)):
                image = Image.open(str(image))
            
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be PIL Image or path to image")
            
            # Convert to RGB if needed
            if image.mode != 'RGB' and self.config.to_rgb:
                image = image.convert('RGB')
            
            # Apply model-specific processing
            if self.model_type == 'ViT':
                inputs = self.processor(
                    images=image,
                    return_tensors=return_tensors
                )
            elif self.model_type == 'CLIP':
                inputs = self.processor(
                    images=image,
                    return_tensors=return_tensors,
                    padding=True
                )
            
            # Move to appropriate device
            return {k: v.to(self.device) for k, v in inputs.items()}
            
        except Exception as e:
            logging.error(f"Image preprocessing failed: {str(e)}")
            raise

    def extract_features(
        self,
        image: Union[Image.Image, str, Path],
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features from image with support for multiple feature types.
        """
        try:
            inputs = self.preprocess_image(image)
            
            # Extract features based on model type
            if self.model_type == 'ViT':
                features = self._extract_vit_features(inputs)
            elif self.model_type == 'CLIP':
                features = self._extract_clip_features(inputs)
            
            if return_dict:
                return {
                    'features': features,
                    'input_shape': tuple(inputs['pixel_values'].shape),
                    'model_type': self.model_type
                }
            return features
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {str(e)}")
            raise

    def _extract_vit_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features using ViT model"""
        # This would typically involve running the actual ViT model
        # For now, we'll return the processed inputs
        return inputs['pixel_values']

    def _extract_clip_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features using CLIP model"""
        # This would typically involve running the actual CLIP model
        # For now, we'll return the processed inputs
        return inputs['pixel_values']

    @staticmethod
    def validate_image(image: Image.Image) -> bool:
        """
        Validate image format and content.
        """
        try:
            # Check image mode
            if image.mode not in {'RGB', 'RGBA', 'L'}:
                return False
            
            # Check image size
            if any(dim <= 0 for dim in image.size):
                return False
            
            # Check for corrupt image
            image.verify()
            return True
            
        except Exception:
            return False 