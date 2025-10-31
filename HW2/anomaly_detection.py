import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
import os

# ===========================
# CONFIGURATION
# ===========================
class Config:
    # Paths
    train_dir = './Dataset/train'
    test_dir = './Dataset/test'
    output_csv = 'submission.csv'
    
    # Training
    img_size = 224
    batch_size = 32
    num_epochs = 100
    lr = 0.0001
    
    # Feature extraction
    feature_layers = [1, 2, 3]  # Which encoder layers to use
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===========================
# DATASET
# ===========================
class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        
        for item in self.root_dir.rglob('*'):
            if item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                self.images.append(item)
        
        self.images = sorted(self.images)
        print(f"Found {len(self.images)} images in {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, str(img_path.name)

# ===========================
# FEATURE EXTRACTOR (Train from Scratch)
# ===========================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Build a deeper network for better features
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.features = {}
    
    def forward(self, x):
        self.features = {}
        
        x1 = self.layer1(x)
        self.features['layer1'] = x1
        
        x2 = self.layer2(x1)
        self.features['layer2'] = x2
        
        x3 = self.layer3(x2)
        self.features['layer3'] = x3
        
        x4 = self.layer4(x3)
        self.features['layer4'] = x4
        
        # Upsample for reconstruction
        x_up = F.interpolate(x4, size=(7, 7), mode='bilinear')
        recon = self.decoder(x_up)
        
        return recon, x4.flatten(1)

# ===========================
# TRAINING
# ===========================
def train_model(model, train_loader, config):
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for imgs, _ in pbar:
            imgs = imgs.to(config.device)
            
            optimizer.zero_grad()
            recon, features = model(imgs)
            
            # Multi-scale reconstruction loss
            loss = F.mse_loss(recon, imgs)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Training completed. Best loss: {best_loss:.6f}')
    return model

# ===========================
# MAHALANOBIS DISTANCE SCORER
# ===========================
class MahalanobisScorer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.train_features = []
        self.means = {}
        self.inv_covs = {}
        
    def extract_training_features(self, train_loader):
        """Extract features from all training samples"""
        self.model.eval()
        
        all_features = {f'layer{i}': [] for i in self.config.feature_layers}
        
        print("Extracting training features...")
        with torch.no_grad():
            for imgs, _ in tqdm(train_loader):
                imgs = imgs.to(self.config.device)
                _, _ = self.model(imgs)
                
                for layer_idx in self.config.feature_layers:
                    layer_name = f'layer{layer_idx}'
                    feat = self.model.features[layer_name]
                    # Global average pooling
                    feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                    all_features[layer_name].append(feat.cpu().numpy())
        
        # Compute mean and covariance for each layer
        print("Computing statistics...")
        for layer_name in all_features.keys():
            features = np.concatenate(all_features[layer_name], axis=0)
            
            # Compute mean
            self.means[layer_name] = np.mean(features, axis=0)
            
            # Compute covariance with shrinkage for numerical stability
            cov_estimator = LedoitWolf()
            cov = cov_estimator.fit(features).covariance_
            self.inv_covs[layer_name] = np.linalg.inv(cov)
            
            print(f"{layer_name}: features shape {features.shape}")
    
    def compute_score(self, img):
        """Compute Mahalanobis distance for test image"""
        self.model.eval()
        
        with torch.no_grad():
            img = img.to(self.config.device)
            recon, _ = self.model(img)
            
            # Reconstruction error
            recon_error = F.mse_loss(recon, img, reduction='none')
            recon_score = recon_error.view(recon_error.size(0), -1).mean(dim=1).cpu().numpy()
            
            # Mahalanobis distance for each layer
            mahal_scores = []
            
            for layer_idx in self.config.feature_layers:
                layer_name = f'layer{layer_idx}'
                feat = self.model.features[layer_name]
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1).cpu().numpy()
                
                # Compute Mahalanobis distance
                for i in range(feat.shape[0]):
                    dist = mahalanobis(feat[i], self.means[layer_name], self.inv_covs[layer_name])
                    mahal_scores.append(dist)
            
            # Average Mahalanobis across layers
            mahal_scores = np.array(mahal_scores).reshape(img.size(0), -1).mean(axis=1)
            
            # Combine reconstruction and Mahalanobis
            final_score = 0.6 * recon_score + 0.4 * mahal_scores
            
            return final_score

# ===========================
# MAIN EXECUTION
# ===========================
def main():
    config = Config()
    
    # Transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
    ])
    
    # Datasets
    train_dataset = MVTecDataset(config.train_dir, transform=train_transform)
    test_dataset = MVTecDataset(config.test_dir, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = FeatureExtractor()
    
    # Train or load
    if os.path.exists('best_model.pth'):
        print("Loading existing model...")
        model.load_state_dict(torch.load('best_model.pth', map_location=config.device))
        model = model.to(config.device)
    else:
        print("Starting training...")
        model = train_model(model, train_loader, config)
        model.load_state_dict(torch.load('best_model.pth', map_location=config.device))
        model = model.to(config.device)
    
    # Build scorer with Mahalanobis distance
    scorer = MahalanobisScorer(model, config)
    
    # Need to extract features with no augmentation
    train_loader_no_aug = DataLoader(
        MVTecDataset(config.train_dir, transform=test_transform),
        batch_size=config.batch_size, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    scorer.extract_training_features(train_loader_no_aug)
    
    # Compute anomaly scores
    print("\nComputing test scores...")
    all_scores = []
    all_names = []
    
    for imgs, names in tqdm(test_loader):
        scores = scorer.compute_score(imgs)
        all_scores.extend(scores)
        all_names.extend(names)
    
    all_scores = np.array(all_scores)
    
    # Normalize to [0, 1]
    scores_normalized = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min() + 1e-8)
    
    # Find optimal threshold using multiple strategies
    print("\nAnalyzing score distribution...")
    print(f"Score statistics:")
    print(f"  Min: {scores_normalized.min():.4f}")
    print(f"  Max: {scores_normalized.max():.4f}")
    print(f"  Mean: {scores_normalized.mean():.4f}")
    print(f"  Median: {np.median(scores_normalized):.4f}")
    print(f"  Std: {scores_normalized.std():.4f}")
    
    # Try different thresholds
    percentiles = [70, 75, 80, 85]
    print("\nThreshold options:")
    for p in percentiles:
        thresh = np.percentile(scores_normalized, p)
        n_anomalies = (scores_normalized > thresh).sum()
        print(f"  {p}th percentile: {thresh:.4f} -> {n_anomalies} anomalies ({100*n_anomalies/len(scores_normalized):.1f}%)")
    
    # Use mean + 0.5*std as threshold (adaptive approach)
    threshold = scores_normalized.mean() + 0.5 * scores_normalized.std()
    # Alternative: use percentile
    # threshold = np.percentile(scores_normalized, 75)
    
    print(f"\nUsing threshold: {threshold:.4f}")
    
    # Convert to binary predictions
    binary_predictions = (scores_normalized > threshold).astype(int)
    
    # Create submission with BINARY predictions (0 or 1)
    results = []
    for name, pred in zip(all_names, binary_predictions):
        idx = int(Path(name).stem)
        results.append({'id': idx, 'prediction': int(pred)})
    
    df = pd.DataFrame(results).sort_values('id')
    df.to_csv(config.output_csv, index=False)
    
    print(f"\nSubmission saved to {config.output_csv}")
    print(f"Anomalies detected: {binary_predictions.sum()} / {len(binary_predictions)} ({100*binary_predictions.mean():.1f}%)")
    
    # Also save continuous scores for analysis
    results_cont = []
    for name, score in zip(all_names, scores_normalized):
        idx = int(Path(name).stem)
        results_cont.append({'id': idx, 'score': float(score)})
    df_cont = pd.DataFrame(results_cont).sort_values('id')
    df_cont.to_csv('scores_continuous.csv', index=False)
    print("Continuous scores saved to scores_continuous.csv for analysis")

if __name__ == '__main__':
    main()