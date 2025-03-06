from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn

import utils.utils as procutils
import utils.img2coords

from models.manual_features import FCN_ManualFeats_Early_Fusion
from models.mlp_mixer_wrapper import MLPMixerWrapper
from models.early_fusion_models import EarlyFusionModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

directory = "input_images"
all_files = procutils.list_all_files(directory)

# -----------
# This transform object is used to detect faces in the image and crop the image to only contain the face
# -----------
img2coords_trm = utils.img2coords.ImageToCoordinatesTransform()

# -----------
# Instantiate model for generating intermediate features
# -----------
fcn_model = FCN_ManualFeats_Early_Fusion()
# print(fcn_model)

# Modify the weights and remove the last layer
fcn_state_dict = torch.load(f"weights/fcn_manual_val_{i}_last_training_weights.pth", map_location=torch.device(device))
fcn_state_dict.pop('layers.45.weight', None)
fcn_state_dict.pop('layers.45.bias', None)
fcn_model.load_state_dict(fcn_state_dict)
fcn_model.to(device)

mlp_mixer_model = MLPMixerWrapper()
mlp_mixer_state_dict = torch.load(f"weights/{i}_mlp_mixer_rgb_yfp_ck_urp_50_last_training_weights.pth", map_location=torch.device(device))
mlp_mixer_model.load_state_dict(mlp_mixer_state_dict)
mlp_mixer_model.mlp_mixer.head = nn.Identity()
mlp_mixer_model.to(device)

# -----------
# Instantiate model for generating intermediate features
# -----------
model = EarlyFusionModel()
model_state_dict = torch.load(f"weights/early_fusion_fcn_rgb_mixer_val_{i}_training_weights_epoch_9.pth")
model.load_state_dict(model_state_dict)
model = model.to(device)

# -----------
# Process each image in input_images
# -----------
for file in all_files:
    # Load the image file
    img = Image.open(file)

    # First crop the image to fit the face
    face_coords = img2coords_trm(img)
    cropped_img = procutils.crop_face(img, face_coords)
    print(cropped_img)

    # Extract the filename for saving the cropped output
    file_name = os.path.splitext(os.path.basename(file))[0]
    cropped_img.save(f"cropped_images/{file_name}.png") # 

    # Pass the image into the model
    