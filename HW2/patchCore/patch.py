import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from tqdm import tqdm
import os
import pandas as pd

# ============= Feature Extractor using Pre-trained ResNet =============
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Use pre-trained ResNet (trained on ImageNet)
        resnet = models.wide_resnet50_2(pretrained=True)
        
        # Extract features from multiple layers
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Extract multi-scale features
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        return [x2, x3]  # Return intermediate features


# ============= Dataset =============
class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, os.path.basename(img_path)


# ============= Build Memory Bank =============
def build_memory_bank(model, dataloader, device='cuda'):
    """Extract features from normal training images"""
    model.eval()
    
    all_features = []
    
    print("Building memory bank from normal training data...")
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)
            features = model(images)
            
            # Concatenate multi-scale features
            for i in range(len(features)):
                # Resize to common size and concatenate
                feat = features[i]
                B, C, H, W = feat.shape
                
                # Adaptive pooling to fixed spatial size
                feat = nn.functional.adaptive_avg_pool2d(feat, (28, 28))
                
                # Reshape: (B, C, H, W) -> (B*H*W, C)
                feat = feat.permute(0, 2, 3, 1).reshape(-1, C)
                all_features.append(feat.cpu().numpy())
    
    # Concatenate all features
    memory_bank = np.concatenate(all_features, axis=0)
    print(f"Memory bank shape: {memory_bank.shape}")
    
    # Subsample for efficiency (optional but recommended)
    if memory_bank.shape[0] > 100000:
        print(f"Subsampling memory bank from {memory_bank.shape[0]} to 100000...")
        indices = np.random.choice(memory_bank.shape[0], 100000, replace=False)
        memory_bank = memory_bank[indices]
    
    # Dimensionality reduction for speed (optional)
    if memory_bank.shape[1] > 512:
        print(f"Reducing dimensions from {memory_bank.shape[1]} to 512...")
        reducer = SparseRandomProjection(n_components=512, random_state=42)
        memory_bank = reducer.fit_transform(memory_bank)
    else:
        reducer = None
    
    return memory_bank, reducer


# ============= Compute Anomaly Scores =============
def compute_anomaly_scores(model, test_loader, memory_bank, reducer, k=9, device='cuda'):
    """Compute anomaly scores using k-NN distance to memory bank"""
    model.eval()
    
    # Build k-NN index
    print(f"Building k-NN index (k={k})...")
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='auto', n_jobs=-1)
    knn.fit(memory_bank)
    
    anomaly_scores = []
    image_names = []
    
    print("Computing anomaly scores...")
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            features = model(images)
            
            # Process each image
            for img_idx in range(images.shape[0]):
                patch_scores = []
                
                # Extract features for this image
                for feat in features:
                    B, C, H, W = feat.shape
                    feat_img = feat[img_idx:img_idx+1]
                    
                    # Adaptive pooling
                    feat_img = nn.functional.adaptive_avg_pool2d(feat_img, (28, 28))
                    
                    # Reshape to patches
                    feat_patches = feat_img.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
                    
                    # Apply dimensionality reduction if used
                    if reducer is not None:
                        feat_patches = reducer.transform(feat_patches)
                    
                    # Compute k-NN distances
                    distances, _ = knn.kneighbors(feat_patches)
                    
                    # Anomaly score: mean of k-NN distances
                    patch_scores.extend(np.mean(distances, axis=1))
                
                # Image-level score: take 90th percentile of patch scores
                image_score = np.percentile(patch_scores, 90)
                
                anomaly_scores.append(image_score)
            
            image_names.extend(names)
    
    return anomaly_scores, image_names


# ============= Main Pipeline =============
def main():
    # Configuration
    IMG_SIZE = 224  # ResNet standard size
    BATCH_SIZE = 32
    K_NEIGHBORS = 9
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    TRAIN_DIR = '../Dataset/train'
    TEST_DIR = '../Dataset/test'
    OUTPUT_CSV = 'submission.csv'
    
    print("="*60)
    print("PATCHCORE ANOMALY DETECTION")
    print("="*60)
    
    # ImageNet normalization (required for pre-trained ResNet)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = MVTecDataset(TRAIN_DIR, transform=transform)
    test_dataset = MVTecDataset(TEST_DIR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    print(f"\nDataset Info:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Device: {DEVICE}")
    
    # Load pre-trained feature extractor
    print("\n" + "="*60)
    print("LOADING PRE-TRAINED FEATURE EXTRACTOR")
    print("="*60)
    model = FeatureExtractor().to(DEVICE)
    model.eval()
    print("✓ Wide ResNet-50 loaded (ImageNet pre-trained)")
    
    # Build memory bank
    print("\n" + "="*60)
    print("BUILDING MEMORY BANK")
    print("="*60)
    memory_bank, reducer = build_memory_bank(model, train_loader, device=DEVICE)
    
    # Compute anomaly scores
    print("\n" + "="*60)
    print("COMPUTING ANOMALY SCORES")
    print("="*60)
    anomaly_scores, image_names = compute_anomaly_scores(
        model, test_loader, memory_bank, reducer, k=K_NEIGHBORS, device=DEVICE
    )
    
    # Create submission
    print("\n" + "="*60)
    print("CREATING SUBMISSION")
    print("="*60)
    
    results = []
    for name, score in zip(image_names, anomaly_scores):
        img_id = int(name.split('.')[0])
        results.append({'id': img_id, 'prediction': score})
    
    results_df = pd.DataFrame(results).sort_values('id')
    
    # Normalize scores to [0, 1]
    min_score = results_df['prediction'].min()
    max_score = results_df['prediction'].max()
    results_df['prediction'] = (results_df['prediction'] - min_score) / (max_score - min_score)
    
    # Save
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✓ Submission saved to {OUTPUT_CSV}")
    print(f"\nScore Statistics:")
    print(results_df['prediction'].describe())
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()