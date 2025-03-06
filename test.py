from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn

import utils.utils as procutils
import utils.img2coords
import utils.img2feats

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
img2feats = utils.img2feats.ImageToManualFeaturesTransform()

# -----------
# Instantiate model for generating intermediate features
# -----------
fcn_model = FCN_ManualFeats_Early_Fusion()
# print(fcn_model)

# Modify the weights and remove the last layer
fcn_state_dict = torch.load(f"weights/fcn_manual_val_0_last_training_weights.pth", map_location=torch.device(device))
fcn_state_dict.pop('layers.45.weight', None)
fcn_state_dict.pop('layers.45.bias', None)
fcn_model.load_state_dict(fcn_state_dict)
fcn_model.to(device)
fcn_model.eval()

mlp_mixer_model = MLPMixerWrapper()
mlp_mixer_state_dict = torch.load(f"weights/0_mlp_mixer_rgb_yfp_ck_urp_50_last_training_weights.pth", map_location=torch.device(device))
mlp_mixer_model.load_state_dict(mlp_mixer_state_dict)
mlp_mixer_model.mlp_mixer.head = nn.Identity()
mlp_mixer_model.to(device).eval()

# -----------
# Instantiate model for generating intermediate features
# -----------
model = EarlyFusionModel()
model_state_dict = torch.load(f"weights/early_fusion_fcn_rgb_mixer_val_0_training_weights_epoch_9.pth", map_location=torch.device(device))
model.load_state_dict(model_state_dict)
model = model.to(device).eval()
results_list = []

# -----------
# Process each image in input_images
# -----------
for file in all_files:
    # Load the image file
    img = Image.open(file)

    # First crop the image to fit the face
    face_coords = img2coords_trm(img)
    try:
        cropped_img = procutils.crop_face(img, face_coords, filename=os.path.splitext(os.path.basename(file))[0])
    except Exception as e:
        results_list.append(f"No face detected by Mediapipe for {file}")

    trfm_img_for_rgb = procutils.preprocess_img(cropped_img)

    # -----------
    # Get Manual Feature embeddings
    # -----------
    feats_tensor = img2feats(cropped_img)
    feats_tensor = torch.tensor(feats_tensor.clone().detach(), dtype=torch.float32).unsqueeze(0)
    manual_feat_embedding = fcn_model(feats_tensor)

    # -----------
    # Get RGB embeddings
    # -----------
    batch_size, number_crops, c, h, w = trfm_img_for_rgb.size()
    images = trfm_img_for_rgb.view(-1, c, h, w).to(device)
    rgb_out = mlp_mixer_model(images)
    rgb_out = rgb_out.view(batch_size, number_crops, -1).mean(1)

    # -----------
    # Concatenate embeddings and feed through fusion model
    # -----------
    combined_tensor = torch.cat((manual_feat_embedding, rgb_out), dim=1)
    output = model(combined_tensor)
    _, predicted = torch.max(output, 1)
    prediction = "Palsy" if predicted == 1 else "Healthy"
    results_list.append(f"Prediction for {file}: {prediction}")

for i in range(len(results_list)):
    print(results_list[i])