import torch
import torchvision.transforms as transforms

def preprocess_img(img):
    test_transforms=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # Convert crops to tensor
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))  # Normalize crops
    ])
    transformed_img = test_transforms(img)
    return transformed_img