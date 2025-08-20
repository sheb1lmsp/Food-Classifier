
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision import transforms
from torch import nn
from typing import Tuple

def get_model(out_features: int) -> Tuple[nn.Module, transforms.Compose]:
    """
    Create a MobileNetV2 model with a modified classification head for transfer learning.

    Args:
        out_features (int): Number of output classes for the classification head.

    Returns:
        nn.Module: Modified MobileNetV2 model with frozen parameters and a new classification head.
        transforms.Compose: MobileNetV2 automatic transform.
    """
    # Load pre-trained MobileNetV2 model
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classification head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=1280, out_features=out_features, bias=True)
    )

    # Create MobileNetV2 transform
    transform = MobileNet_V2_Weights.DEFAULT.transforms()

    return model, transform
