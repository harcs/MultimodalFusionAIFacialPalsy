'''
This is a custom transform to be applied to a PyTorch image dataset, transforming
each image to a list of xy facial landmark coordinates
'''
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class ImageToCoordinatesTransform(object):
    def __init__(self):
        pass

    def __call__(self, img):
        width, height = img.size
        coordinates = self.image_to_landmark_coords(img)
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
        coordinates[:, 0] *= width
        coordinates[:, 1] *= height
        return coordinates
    
    def __repr__(self):
        return "3D facial landmarks extracted from image"

    def image_to_landmark_coords(self, image):
        img_np = np.asarray(image)
        if len(img_np.shape) == 2:  # Grayscale images are 2D
            # Convert grayscale to RGB by stacking the single channel three times
            img_np = np.stack([img_np] * 3, axis=-1)
        # Convert the PIL image to a MediaPipe Image
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

        # Get the landmarks of the image
        detection_result = self.run_inference(image)

        # Get all coordinates
        # print(detection_result.face_landmarks[0])
        xy_coords = [(face_landmark.x, face_landmark.y) for face_landmark in detection_result.face_landmarks[0]]
        xy_coords = torch.tensor(xy_coords)

        return xy_coords

    def run_inference(self, image):
        # STEP 1: Create an FaceLandmarker object.
        base_options = python.BaseOptions(model_asset_path='utils/face_landmarker_v2_with_blendshapes.task', delegate= "GPU")
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True, num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)

        # STEP 2: Detect face landmarks from the input image.
        detection_result = detector.detect(image)
        return detection_result