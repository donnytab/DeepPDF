import os
import PIL
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

# PIXEL_DIMENSION_WIDTH = 1828
# PIXEL_DIMENSION_HEIGHT = 1306
PIXEL_DIMENSION_WIDTH = 360
PIXEL_DIMENSION_HEIGHT = 240
IMG_RESOURCE_PATH = os.getcwd() + "/res/"

def load_image():

    print("Loading image data...")

    img_input = []
    sample_size = 0

    data_folder = os.path.join(os.getcwd(), IMG_RESOURCE_PATH)
    for files in os.listdir(data_folder):
        print("file: ", str(files))
        # PDF files not found
        if files.find(".pdf") == -1:
            continue
        files = os.path.join(data_folder, files)
        images = convert_from_path(str(files)).pop(0)
        img = images.resize((PIXEL_DIMENSION_WIDTH, PIXEL_DIMENSION_HEIGHT),PIL.Image.ANTIALIAS)
        print(img)
        print(list(img.getdata())[0])
        img_input.append(reshape_image(img))
        sample_size += 1
    return img_input, sample_size
        # print(len(reshape_image(img)))

    # return img_input

def reshape_image(temp_image):
    # reshaped = np.array(temp_image.getdata()).reshape(temp_image.size[0], temp_image.size[1], 1)
    reshaped = np.array(temp_image.getdata())
    reshaped = reshaped.tolist()

    return reshaped