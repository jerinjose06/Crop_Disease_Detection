# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvBlock(nn.Module):
    """Convolutional Block with Conv2d, BatchNorm, ReLU, and optional MaxPool."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, pool=True, pool_size=2):
        super(ConvBlock, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if pool:
            layers.append(nn.MaxPool2d(pool_size))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual Block for deeper networks."""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class CropDiseaseCNN(nn.Module):
    """
    Custom CNN Architecture for Crop Disease Classification.

    Architecture:
    - 6 Convolutional blocks with increasing filters
    - Residual connections for better gradient flow
    - Global Average Pooling
    - Fully connected classifier with dropout
    """

    def __init__(self, num_classes=config.NUM_CLASSES, dropout_rate=config.DROPOUT_RATE):
        super(CropDiseaseCNN, self).__init__()

        # Feature Extractor
        # Block 1: 3 -> 32 channels, 224x224 -> 112x112
        self.conv_block1 = ConvBlock(3, 32, pool=True)

        # Block 2: 32 -> 64 channels, 112x112 -> 56x56
        self.conv_block2 = ConvBlock(32, 64, pool=True)
        self.res_block1 = ResidualBlock(64)

        # Block 3: 64 -> 128 channels, 56x56 -> 28x28
        self.conv_block3 = ConvBlock(64, 128, pool=True)
        self.res_block2 = ResidualBlock(128)

        # Block 4: 128 -> 256 channels, 28x28 -> 14x14
        self.conv_block4 = ConvBlock(128, 256, pool=True)
        self.res_block3 = ResidualBlock(256)

        # Block 5: 256 -> 512 channels, 14x14 -> 7x7
        self.conv_block5 = ConvBlock(256, 512, pool=True)
        self.res_block4 = ResidualBlock(512)

        # Block 6: Additional conv without pooling
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),

            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.res_block1(x)
        x = self.conv_block3(x)
        x = self.res_block2(x)
        x = self.conv_block4(x)
        x = self.res_block3(x)
        x = self.conv_block5(x)
        x = self.res_block4(x)
        x = self.conv_block6(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)

        # Classification
        x = self.classifier(x)

        return x


class CropDiseaseCNNLight(nn.Module):
    """
    Lighter CNN Architecture for systems with limited resources.
    """

    def __init__(self, num_classes=config.NUM_CLASSES, dropout_rate=config.DROPOUT_RATE):
        super(CropDiseaseCNNLight, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32, 224 -> 112
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 -> 64, 112 -> 56
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 -> 128, 56 -> 28
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128 -> 256, 28 -> 14
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: 256 -> 512, 14 -> 7
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x


def get_model(model_type='full', num_classes=config.NUM_CLASSES):
    """
    Factory function to create model.

    Args:
        model_type: 'full' for CropDiseaseCNN, 'light' for CropDiseaseCNNLight
        num_classes: Number of output classes
    """
    if model_type == 'full':
        model = CropDiseaseCNN(num_classes=num_classes)
    elif model_type == 'light':
        model = CropDiseaseCNNLight(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def count_parameters(model):
    """Count trainable parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_summary(model, input_size=(1, 3, 224, 224)):
    """Print model summary."""
    print(f"\n{'='*60}")
    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"{'='*60}")
    print(model)

    total, trainable = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"{'='*60}\n")

    # Switch to eval mode so BatchNorm works with batch_size=1
    model.eval()

    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"{'='*60}\n")

    # Switch back to train mode for actual training
    model.train()