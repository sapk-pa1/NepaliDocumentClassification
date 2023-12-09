from keras.applications.resnet50 import preprocess_input
import numpy as np 
from PIL import Image



def process_img(image) :    
    """Process the image from the preprocess input from the resnet50 

    Args:
        img (_type_): unprocessed image 

    Returns:
        img: processed image for using with the model
    """
    # Preprocess the image using ResNet50's preprocess_input
    # If the input is a file path, open the image
    if isinstance(image, str):
        # If the input is a file path, open the image
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        # If the input is a NumPy array, create a PIL Image
        img = Image.fromarray(image.astype('uint8'), 'RGB')
    else:
        raise ValueError("Unsupported image type. Please provide a file path or a NumPy array.")

    img = img.resize((500, 500))
    img_array = np.array(img)
    img_array = img_array[np.newaxis, ...]
    img_array = preprocess_input(img_array)
    return img_array