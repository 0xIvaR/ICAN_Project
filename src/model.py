"""
ICAN - Intelligent Condition-Adaptive Network
Multi-task model for gender classification and face recognition under adverse conditions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import Config

class AttentionModule(nn.Module):
    """Spatial attention module to focus on important facial features"""
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class ConditionAdaptiveBlock(nn.Module):
    """Adaptive block to handle different visual conditions"""
    def __init__(self, in_features, hidden_features):
        super(ConditionAdaptiveBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(in_features)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual  # Residual connection
        x = self.layer_norm(x)
        return x

class ICAN(nn.Module):
    """
    Intelligent Condition-Adaptive Network
    Multi-task learning model for gender classification and face recognition
    """
    def __init__(self, num_identity_classes, backbone_name='efficientnet_b3', pretrained=True):
        super(ICAN, self).__init__()
        
        self.backbone_name = backbone_name
        self.num_identity_classes = num_identity_classes
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            backbone_output = self.backbone(dummy_input)
            if len(backbone_output.shape) == 4:  # Conv output (B, C, H, W)
                self.feature_dim = backbone_output.shape[1]
                self.spatial_dims = backbone_output.shape[2:]
            else:  # Already flattened
                self.feature_dim = backbone_output.shape[1]
                self.spatial_dims = None
        
        # Attention module for feature enhancement
        if self.spatial_dims is not None:
            self.attention = AttentionModule(self.feature_dim)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.attention = None
            self.global_pool = None
        
        # Condition-adaptive processing
        self.adaptive_block1 = ConditionAdaptiveBlock(self.feature_dim, self.feature_dim // 2)
        self.adaptive_block2 = ConditionAdaptiveBlock(self.feature_dim, self.feature_dim // 2)
        
        # Shared feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512)
        )
        
        # Task-specific heads
        # Gender classification head (binary)
        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, Config.NUM_CLASSES_GENDER)
        )
        
        # Face recognition head (multi-class)
        self.identity_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_identity_classes)
        )
        
        # Feature normalization for better convergence
        self.feature_norm = nn.BatchNorm1d(512)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply attention if spatial dimensions exist
        if self.attention is not None and len(features.shape) == 4:
            features = self.attention(features)
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)
        elif len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
        
        # Apply condition-adaptive processing
        features = self.adaptive_block1(features)
        features = self.adaptive_block2(features)
        
        # Shared feature processing
        shared_features = self.shared_fc(features)
        shared_features = self.feature_norm(shared_features)
        
        # Task-specific predictions
        gender_logits = self.gender_head(shared_features)
        identity_logits = self.identity_head(shared_features)
        
        if return_features:
            return gender_logits, identity_logits, shared_features
        
        return gender_logits, identity_logits
    
    def get_embedding(self, x):
        """Get feature embeddings for similarity comparisons"""
        with torch.no_grad():
            _, _, embeddings = self.forward(x, return_features=True)
            # L2 normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

class ICANLoss(nn.Module):
    """
    Multi-task loss function for ICAN
    Combines gender classification loss and face recognition loss
    """
    def __init__(self, gender_weight=0.3, identity_weight=0.7, label_smoothing=0.1):
        super(ICANLoss, self).__init__()
        self.gender_weight = gender_weight
        self.identity_weight = identity_weight
        
        # Loss functions
        self.gender_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.identity_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, gender_logits, identity_logits, gender_targets, identity_targets):
        gender_loss = self.gender_criterion(gender_logits, gender_targets)
        identity_loss = self.identity_criterion(identity_logits, identity_targets)
        
        total_loss = (self.gender_weight * gender_loss + 
                     self.identity_weight * identity_loss)
        
        return total_loss, gender_loss, identity_loss

def create_ican_model(num_identity_classes, backbone_name=None):
    """
    Factory function to create ICAN model
    """
    if backbone_name is None:
        backbone_name = Config.BACKBONE
    
    model = ICAN(
        num_identity_classes=num_identity_classes,
        backbone_name=backbone_name,
        pretrained=Config.PRETRAINED
    )
    
    return model

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model
    model = create_ican_model(num_identity_classes=100)
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Backbone: {Config.BACKBONE}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    gender_out, identity_out = model(dummy_input)
    print(f"Gender output shape: {gender_out.shape}")
    print(f"Identity output shape: {identity_out.shape}")
    
    # Test loss
    criterion = ICANLoss()
    gender_targets = torch.randint(0, 2, (4,))
    identity_targets = torch.randint(0, 100, (4,))
    loss, gender_loss, identity_loss = criterion(gender_out, identity_out, gender_targets, identity_targets)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Gender loss: {gender_loss.item():.4f}")
    print(f"Identity loss: {identity_loss.item():.4f}") 