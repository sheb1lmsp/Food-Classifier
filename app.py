
import gradio as gr
from pathlib import Path
from scripts import predict

# Create title and description strings.
title = "FoodClassifier Mini"
description = "An MobileNet_V2 feature extractor computer vision model to classify images of food as apple_pie, donuts, french_fries, hamburger, hot_dog, ice_cream, pizza, steak, sushi and tacos."

# Create example image list for gradio.
example_list = [str(x) for x in list(Path('examples').glob('*.jpg'))]

# Creating gradio interface demo.
demo = gr.Interface(
    fn = predict,
    inputs = gr.Image(type='pil'),
    outputs = [
        gr.Label(num_top_classes=5, label='Predictions'),
        gr.Number(label='Prediction time'),
    ],
    examples = example_list,
    title = title,
    description = description
)

# Launch the gradio demo!.
demo.launch()
