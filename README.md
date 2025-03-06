# MultimodalAIFacialPalsy
# How To Use
1. ```pip install -r requirements``` to install dependencies
2. Download model weights from this Google Drive: https://drive.google.com/drive/u/0/folders/1Pr5BHciXvLqCVG6TmK0I2H7XPhL96caN, then create a folder named "weights" and add the weights into that folder
2. Add your test images into the input_images folder
3. Run ```python3 test.py``` to see classification outputs for each image.
4. The cropped_images folder contains intermediate outputs which can be checked for debugging (landmark coordinate predictions are used to crop the picture to fit the face)

# Sample Output for Provided Images
The images in this repository were taken from the internet. These are the results:
```
Prediction for input_images\palsy1.jpg: Palsy

Prediction for input_images\palsy10.jpg: Palsy

Prediction for input_images\palsy2.jpg: Healthy

Prediction for input_images\palsy3.png: Palsy

Prediction for input_images\palsy4.jpg: Palsy

Prediction for input_images\palsy5.jpg: Palsy

Prediction for input_images\palsy6.jpeg: Palsy

Prediction for input_images\palsy7.jpg: Palsy

No face detected by Mediapipe for input_images\palsy8.png

Prediction for input_images\palsy9.jpg: Palsy
```

# Extra Notes on Input Images
It seems that the MediaPipe model struggles to detect faces in images where the camera is too zoomed in on the face. For example, it is unable to detect a face for palsy8, but is successful for palsy7 (identical to palsy7, but the image is expanded outwards)
