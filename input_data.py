import os
import numpy as np
from PIL import Image

IMG_RESOURCE_PATH = os.getcwd() + "/res/"

def load_image():

    print("Loading image data...")

    img_input = []

    data_folder = os.path.join(os.getcwd(), IMG_RESOURCE_PATH)
    for images in os.listdir(data_folder):
        images = os.path.join(data_folder, images)
        img = Image.open(images)
        print(img)
        print(list(img.getdata())[0])
        img_input.append(reshape_image(img))
        return reshape_image(img)
        # print(len(reshape_image(img)))

    # return img_input

def reshape_image(temp_image):
    # reshaped = np.array(temp_image.getdata()).reshape(temp_image.size[0], temp_image.size[1], 1)
    reshaped = np.array(temp_image.getdata())
    reshaped = reshaped.tolist()

    return reshaped


load_image()