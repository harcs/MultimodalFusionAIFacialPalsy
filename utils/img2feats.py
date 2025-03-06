'''
This is a custom transform to be applied to a PyTorch image dataset, transforming
each image to a list of manual features
'''
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math_utils
import cv2
from utils.coordinate_info import mesh_annotations
from sklearn.metrics.pairwise import euclidean_distances

class ImageToManualFeaturesTransform(object):
    def __init__(self):
        self.mesh_annotations = mesh_annotations
        self.coordinates = {}

    def __call__(self, img):
        width, height = img.size
        self.width, self.height = width, height
        features = self.image_to_features(img)
        return features
    
    def __repr__(self):
        return "3D facial landmarks extracted from image"
    
    def run_inference(self, image):
        # STEP 1: Create an FaceLandmarker object.
        base_options = python.BaseOptions(model_asset_path='utils/face_landmarker_v2_with_blendshapes.task', delegate= "CPU")
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True, num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)

        # STEP 2: Detect face landmarks from the input image.
        detection_result = detector.detect(image)
        return detection_result

    def image_to_features(self, image):
        # Convert the PIL image to a MediaPipe Image
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))

        # Get the landmarks of the image
        detection_result = self.run_inference(image)
        detection_result = detection_result.face_landmarks[0]

        # --------------
        # Get the coordinates for each feature group
        # --------------
        for feature in self.mesh_annotations.keys():
            coords_obj = detection_result[self.mesh_annotations[feature][0]]
            self.coordinates[feature] = [coords_obj.x, coords_obj.y]

        # --------------
        # Calculate the rotation matrix
        # --------------
        angle_for_rotation = math_utils.calculate_angle(self.coordinates[48], self.coordinates[49])
        rotation_matrix = cv2.getRotationMatrix2D(np.array(self.coordinates[48]), angle_for_rotation, 1)

        # --------------
        # Rotate all coordinates
        # --------------
        for feature in self.mesh_annotations.keys():
            augmented_coords = np.array([self.coordinates[feature]])
            rotated_coords = cv2.transform(augmented_coords[None, :, :], rotation_matrix)[0]
            self.coordinates[feature] = rotated_coords[0]

        # ----------------------------------------------------------------------------
        # Calculate manual features
        # ----------------------------------------------------------------------------
        # First calculate distances
        self.calculate_a_k_distances()
        self.calculate_l_q_distances()
        self.calculate_r_x_distances()

        # Now calculate ratios and angles
        eyebrow_feats = self.calculate_eyebrow_features()
        eye_feats = self.calculate_eye_features()
        mouth_feats = self.calculate_mouth_features()
        nose_feats = self.calculate_nose_features()
        combined_feats = self.calculate_combined_features()
        feature_tensor = eyebrow_feats + eye_feats + mouth_feats + nose_feats + combined_feats
        feature_tensor = torch.tensor(feature_tensor)
        return feature_tensor

    def calculate_a_k_distances(self):
        self.dist_a = abs(self.coordinates[48][0] - self.coordinates[49][0])
        self.dist_bl = abs(self.coordinates[10][0] - self.coordinates[13][0])
        self.dist_br = abs(self.coordinates[16][0] - self.coordinates[19][0])
        self.dist_c = euclidean_distances([self.coordinates[37]], [self.coordinates[50]])
        self.dist_d = abs(self.coordinates[48][0] - self.coordinates[10][0])
        self.dist_e = abs(self.coordinates[19][0] - self.coordinates[49][0])
        self.dist_f = euclidean_distances([self.coordinates[10]], [self.coordinates[37]])
        self.dist_g = euclidean_distances([self.coordinates[19]], [self.coordinates[37]])
        self.dist_h = euclidean_distances([self.coordinates[10]], [self.coordinates[23]])
        self.dist_i = euclidean_distances([self.coordinates[19]], [self.coordinates[27]])
        self.dist_j = euclidean_distances([self.coordinates[23]], [self.coordinates[37]])
        self.dist_k = euclidean_distances([self.coordinates[27]], [self.coordinates[37]])

    def calculate_l_q_distances(self):
        self.dist_l = sum([self.coordinates[0][1], self.coordinates[1][1], self.coordinates[2][1], self.coordinates[3][1], self.coordinates[4][1]]) / 5
        self.dist_m = sum([self.coordinates[5][1], self.coordinates[6][1], self.coordinates[7][1], self.coordinates[8][1], self.coordinates[9][1]]) / 5
        self.dist_nl = euclidean_distances([self.coordinates[11]], [self.coordinates[15]])
        self.dist_nr = euclidean_distances([self.coordinates[12]], [self.coordinates[14]])
        self.dist_n = (self.dist_nl + self.dist_nr) / 2
        self.dist_ol = euclidean_distances([self.coordinates[17]], [self.coordinates[21]])
        self.dist_or = euclidean_distances([self.coordinates[18]], [self.coordinates[20]])
        self.dist_o = (self.dist_ol + self.dist_or) / 2
        self.dist_pl = euclidean_distances([self.coordinates[29]], [self.coordinates[39]])
        self.dist_pu = euclidean_distances([self.coordinates[30]], [self.coordinates[38]])
        self.dist_qu = euclidean_distances([self.coordinates[32]], [self.coordinates[36]])
        self.dist_ql = euclidean_distances([self.coordinates[33]], [self.coordinates[35]])

    def calculate_r_x_distances(self):
        self.dist_r = euclidean_distances([self.coordinates[3]], [self.coordinates[37]])
        self.dist_s = euclidean_distances([self.coordinates[6]], [self.coordinates[37]])
        self.dist_t = euclidean_distances([self.coordinates[2]], [self.coordinates[37]])
        self.dist_u = euclidean_distances([self.coordinates[7]], [self.coordinates[37]])
        self.dist_vl = euclidean_distances([self.coordinates[28]], [self.coordinates[37]])
        self.dist_vr = euclidean_distances([self.coordinates[34]], [self.coordinates[37]])
        self.dist_w = abs(self.coordinates[34][0] - self.coordinates[28][0])
        self.dist_wl = math_utils.calculate_perimeter([self.coordinates[28], self.coordinates[29], self.coordinates[30], self.coordinates[31], self.coordinates[37], self.coordinates[38], self.coordinates[39]], s=0, l=5)
        self.dist_wr = math_utils.calculate_perimeter([self.coordinates[31], self.coordinates[32], self.coordinates[33], self.coordinates[34], self.coordinates[35], self.coordinates[36], self.coordinates[37]], s=0, l=6)
        self.dist_x = euclidean_distances([self.coordinates[31]], [self.coordinates[22]]) # Changed from the original

    def calculate_eyebrow_features(self):
        f0 = math_utils.calculate_angle(self.coordinates[0], self.coordinates[9])
        f1 = math_utils.calculate_angle(self.coordinates[2], self.coordinates[7])
        f2 = math_utils.calculate_angle(self.coordinates[4], self.coordinates[5])
        f3 = max(self.dist_l / self.dist_m, self.dist_m / self.dist_l)
        f4 = math_utils.calculate_slope_m(self.coordinates[0], self.coordinates[9])
        f5 = math_utils.calculate_slope_m(self.coordinates[2], self.coordinates[7])
        f6 = math_utils.calculate_slope_m(self.coordinates[4], self.coordinates[5])
        return [f0, f1, f2, f3, f4, f5, f6]

    def calculate_eye_features(self):
        f7 = math_utils.calculate_angle(self.coordinates[10], self.coordinates[19])
        f8 = max(self.dist_bl / self.dist_br, self.dist_br / self.dist_bl)
        f9 = max(self.dist_d / self.dist_e, self.dist_e / self.dist_d)
        f10 = max(self.dist_h / self.dist_i, self.dist_i / self.dist_h)
        f11 = max(self.dist_n / self.dist_o, self.dist_o / self.dist_n)
        f12 = max(self.dist_nl / self.dist_or, self.dist_or / self.dist_nl)
        f13 = max(self.dist_nr / self.dist_ol, self.dist_ol / self.dist_nr)
        return [f7, f8, f9, f10, f11, f12, f13]
    
    def calculate_mouth_features(self):
        f14 = math_utils.calculate_angle(self.coordinates[28], self.coordinates[34])
        f15 = max(self.dist_f / self.dist_g, self.dist_g / self.dist_f)
        f16 = max(self.dist_pl / self.dist_ql, self.dist_ql / self.dist_pl)
        f17 = max(self.dist_pu / self.dist_qu, self.dist_qu / self.dist_pu)
        f18 = max(self.dist_vl / self.dist_a, self.dist_vr / self.dist_a)
        f19 = max(self.dist_pl / self.dist_w, self.dist_ql / self.dist_w)
        f20 = max(self.dist_pu / self.dist_w, self.dist_qu / self.dist_w)
        f21 = max(self.dist_wl / self.dist_w, self.dist_wr / self.dist_w)
        return [f14, f15, f16, f17, f18, f19, f20, f21]

    def calculate_nose_features(self):
        f22 = math_utils.calculate_angle(self.coordinates[23], self.coordinates[27])
        return [f22]
    
    def calculate_combined_features(self):
        f23 = math_utils.calculate_angle(self.coordinates[22], self.coordinates[37])
        f24 = max(self.dist_j / self.dist_k, self.dist_k / self.dist_j)
        f25 = max(self.dist_t / self.dist_a, self.dist_u / self.dist_a)
        f26 = max(self.dist_r / self.dist_a, self.dist_s / self.dist_a)
        f27 = self.dist_c / self.dist_a
        f28 = self.dist_x / self.dist_a
        return [f23, f24, f25, f26, f27, f28]