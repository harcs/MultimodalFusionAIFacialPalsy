# MultimodalAIFacialPalsy
# How To Use
1. ```pip install -r requirements``` to install dependencies
2. Download model weights from this Google Drive: https://drive.google.com/drive/u/0/folders/1Pr5BHciXvLqCVG6TmK0I2H7XPhL96caN, then create a folder named "weights" and add the weights into that folder
2. Add your test images into the input_images folder
3. Run ```python3 test.py``` to see classification outputs for each image.
4. The cropped_images folder contains intermediate outputs which can be checked for debugging (landmark coordinate predictions are used to crop the picture to fit the face)
