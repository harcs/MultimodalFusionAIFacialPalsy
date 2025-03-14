import os
import torch
import torchvision.transforms as transforms

def preprocess_img(img):
    """
    This function applies a transformation to test images
    """
    test_transforms=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))
    ])
    transformed_img = test_transforms(img)
    print(transformed_img.shape)
    return transformed_img.unsqueeze(0)

def crop_face(img, coords_tensor, filename):
    """
    This function uses provided coordinates to crop the image to fit the subject's face
    """
    try:
        width, height = img.size
        print(img.size)
        left = coords_tensor[127][0].item()
        upper = coords_tensor[10][1].item()
        right = coords_tensor[356][0].item()
        lower = coords_tensor[152][1].item()

        # Define crop rectangle (left, upper, right, lower)
        crop_rectangle = (left, upper, right, lower)

        # Crop the image
        cropped_img = img.crop(crop_rectangle)
        cropped_img.save(f"cropped_images/{filename}.png") # Save the cropped image for debugging
        return cropped_img
    
    except Exception as e:
        return None

def list_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths