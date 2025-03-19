import torch
import torchvision.models as models  # CNN models
import torchvision.transforms as transforms
from PIL import Image
import timm  # for EfficientNet models


# ===============================================
# Feature Extractor using Pre-trained CNN Models
# ===============================================
def extract_features_CNN(frame_path, model_name='squeezenet1_0'):
    # Load a pre-trained model
    # SqueezeNet is one of the smallest in terms of model size
    if model_name == 'squeezenet1_0':  # Parameters = 1.2M
        model = models.squeezenet1_0(pretrained=True)
    if model_name == 'squeezenet1_1':  # 1.25M
        model = models.squeezenet1_1(pretrained=True)
    if model_name == 'mobilenet_v2':  # 3.5M
        models.mobilenet_v2(pretrained=True)
    if model_name == 'alexnet':  # 61M
        models.alexnet(pretrained=True)
    if model_name == 'vgg11':  # 132M
        models.vgg11(pretrained=True)
    if model_name == 'vgg16':  # 138M
        models.vgg16(pretrained=True)
    if model_name == 'vgg19':  # 143M
        models.vgg19(pretrained=True)
    if model_name == 'resnet18':  # 11.7M
        models.resnet18(pretrained=True)
    if model_name == 'resnet50':  # 25.6M
        models.resnet50(pretrained=True)
    if model_name == 'resnet101':  # 44.6M
        models.resnet101(pretrained=True)
    if model_name == 'densenet121':  # 7M
        models.densenet121(pretrained=True)
    if model_name == 'densenet169':  # 14.2 M
        models.densenet169(pretrained=True)
    if model_name == 'densenet201':  # 20 M
        models.densenet201(pretrained=True)
    if model_name == 'inception_v3':  # 27.2M
        models.inception_v3(pretrained=True)
    if model_name == 'efficientnet_b0':  # 5.3M
        model = timm.create_model("efficientnet_b0", pretrained=True)
    if model_name == 'efficientnet_b1':  # 7.8M
        model = timm.create_model("efficientnet_b1", pretrained=True)
    if model_name == 'efficientnet_b2':  # 9.2M
        model = timm.create_model("efficientnet_b2", pretrained=True)
    if model_name == 'efficientnet_b3':  # 12.2 M
        model = timm.create_model("efficientnet_b3", pretrained=True)
    if model_name == 'efficientnet_b4':  # 19.3M
        model = timm.create_model("efficientnet_b4", pretrained=True)
    if model_name == 'efficientnet_b5':  # 30 M
        model = timm.create_model("efficientnet_b4", pretrained=True)
    if model_name == 'efficientnet_b6':  # 43 M
        model = timm.create_model("efficientnet_b4", pretrained=True)
    if model_name == 'efficientnet_b7':  # 66 M
        model = timm.create_model("efficientnet_b4", pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Define a transformation to preprocess the image frame
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to VGG16 input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load an image frame (replace 'frame.jpg' with your frame file path)
    frame = Image.open(frame_path)

    # Preprocess the image frame
    frame = transform(frame)
    frame = frame.unsqueeze(0)  # Add a batch dimension

    # Pass the frame through the VGG16 model to extract features
    with torch.no_grad():
        features = model(frame)

    # The 'features' variable now contains the extracted features for the frame: a tensor (1, 1000)
    # print(features.shape)  # This should print the shape of the feature tensor
    return features


# =======================================================
# Feature Extractor using Pre-trained Transformer Models
# with the help of libraries like:
# Hugging Face's Transformers
# or timm (PyTorch Image Models).
# =======================================================
from transformers import ViTFeatureExtractor, ViTForImageClassification
import requests


# Option 1: Using Hugging Face's Transformers
# # !pip install transformers
def extract_features_Tramsformer_HuggingFace(frame_path, model_name='google/vit-base-patch16-224'):
    # Load feature extractor and model
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)

    # Prepare an Image:
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    # Transform and Make Prediction:
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])
    return inputs


# Option 2: Using Hugging Face's Transformers
# !pip install timm
def extract_features_Tramsformer_timms(frame_path, model_name='vit_base_patch16_224'):
    # Load model
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    # Prepare and Predict:
    # Load and transform an image
    img = Image.open(frame_path)
    img = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        preds = model(img)
        predicted_class_idx = preds[0].argmax(-1).item()
        print("Predicted class:", predicted_class_idx)
