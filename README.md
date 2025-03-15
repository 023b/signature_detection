# Signature Classification Model

A deep learning model for signature verification and classification using PyTorch and a pre-trained VGG16 architecture.

## Overview

This project implements a signature classification system that can identify and verify signatures based on training data. The model uses transfer learning with VGG16 as the base model and is customized for signature recognition.

## Features

- Signature classification and verification
- Transfer learning using pre-trained VGG16 model
- Fine-tuning capabilities for custom signature datasets
- High accuracy prediction with probability scores

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- Pillow (PIL)
- CUDA-compatible GPU (optional, for faster training)

## Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/signature-classifier.git
   cd signature-classifier
   ```

2. Install required packages
   ```
   pip install torch torchvision pillow
   ```

## Dataset Structure

The dataset should be organized in the following structure:
```
sign/
├── Person1/
│   ├── signature1.png
│   ├── signature2.png
│   └── ...
├── Person2/
│   ├── signature1.png
│   ├── signature2.png
│   └── ...
└── ...
```

Where each subdirectory represents a different person's signature class.

## Model Architecture

The model uses a pre-trained VGG16 network with:
- Frozen feature extraction layers (to preserve learned features)
- Modified classifier layers for signature classification
- Input image size of 224×224 pixels

## Usage

### Training the Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

# Set parameters
img_width, img_height = 224, 224
train_folder = "path/to/signature/dataset"
batch_size = 3
num_epochs = 9

# Data preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder(train_folder, transform=data_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, len(train_dataset.classes))

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    exp_lr_scheduler.step()
```

### Classifying a New Signature

```python
from PIL import Image

def classify_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = data_transforms(img).unsqueeze(0)
    
    with torch.no_grad():
        img = img.to(device)
        outputs = model(img)
        probs = nn.functional.softmax(outputs, dim=1)[0]
        _, preds = torch.max(outputs, 1)
    
    class_name = train_dataset.classes[preds.item()]
    probability = probs[preds.item()].item()
    
    print(f'Prediction: {class_name} with probability: {probability*100:.2f}%')

# Example usage
img_path = "path/to/test/signature.png"
classify_image(img_path)
```

## Saving and Loading the Model

### Save the trained model
```python
torch.save(model.state_dict(), 'signature_classifier.pth')
```

### Load the model for inference
```python
model = models.vgg16(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, num_classes)  # num_classes = number of signature classes
model.load_state_dict(torch.load('signature_classifier.pth'))
model.eval()
```

## Performance Optimization

- Increase `batch_size` for faster training (if memory allows)
- Increase `num_epochs` for better accuracy
- Use data augmentation for more robust training
- Experiment with different learning rates

## Future Improvements

- Implement validation split to monitor overfitting
- Add data augmentation to improve model generalization
- Incorporate a Siamese network for direct signature comparison
- Add forgery detection capabilities
- Implement a web/mobile interface for easy use

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- VGG team for the pre-trained model architecture
