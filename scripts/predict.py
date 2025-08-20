
import torch
import PIL
from pathlib import Path
from timeit import default_timer as timer
from typing import Tuple, Dict
from create_model import get_model

# Setting device to cpu in order to avoid issues when deploying to gradio.
device = 'cpu'

# Retrieve and store selected classes from the class_names.txt file.
class_names_file_name = Path('../class_names.txt')
with open(class_names_file_name, 'r') as f:
    selected_classes = [x.strip() for x in f.readlines()]

# Initialize the path to the model.
model_path = Path('../model/mobilenet_v2_on_food101.pth')

def predict(image: PIL.Image) -> Tuple[Dict[str,float], float]:
    """
    This function loads the model, inference the image and predict the probability for each class.

    Args:
        image (PIL.Image): The input image that the model has to predict on.

    Returns:
        Tuple[Dict[str, float], float]: The dictionary of classes along with curresponding probabilities and the prediction time.
    """
    # Starting the time.
    start = timer()

    # Loading the model and automatic transform.
    model, transform = get_model(out_features = len(selected_classes))
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)

    # Transform the image.
    transformed_image = transform(image).to(device).unsqueeze(dim=0)

    # Put the model in evaluation mode.
    model.to(device).eval()

    with torch.no_grad():
        # Forward pass.
        y_logit = model(transformed_image)

        # Calculate the probability.
        y_prob = torch.softmax(y_logit, dim=1)

    # Calculate each probability with respect to each class.
    pred_labels_and_probs = {selected_classes[i] : y_prob[0][i].item() for i in range(len(selected_classes))}

    # Calculate the prediction time.
    end = timer()
    pred_time = end - start

    return pred_labels_and_probs, pred_time 
    
