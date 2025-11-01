import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage.feature import hog
from skimage.color import rgb2gray
import warnings
import os
import pickle
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ============================================
# APPROACH: Deep Embedded Clustering + Per-Class Anomaly Detection
# ============================================

class AutoEncoder(nn.Module):
    """Autoencoder for learning embeddings + reconstruction"""
    def __init__(self, latent_dim=128, img_size=224):
        super().__init__()
        
        # Encoder - works for any image size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # /2
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # /4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # /8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # /16
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), # /32
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim)
        )
        
        # Calculate the size after downsampling
        self.feature_size = img_size // 32  # After 5 stride-2 convs
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.feature_size * self.feature_size),
            nn.ReLU(),
            nn.Unflatten(1, (256, self.feature_size, self.feature_size)),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1), # *2
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # *2
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # *2
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # *2
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # *2
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class ImageDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.img_paths[idx]

# ============================================
# STEP 1: Train Autoencoder on All Training Data
# ============================================

def train_autoencoder(train_paths, epochs=30, batch_size=32, img_size=128):
    """Train autoencoder to learn good embeddings"""
    print(f"Training autoencoder on {len(train_paths)} images...")
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(train_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(latent_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for imgs, _ in pbar:
                imgs = imgs.to(device)
                
                optimizer.zero_grad()
                recon, _ = model(imgs)
                loss = criterion(recon, imgs)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")            
    
    return model

# ============================================
# STEP 2: Extract Embeddings and Cluster
# ============================================

def extract_embeddings(model, img_paths, batch_size=32, img_size=128):
    """Extract embeddings for all images"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(img_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    embeddings = []
    paths = []
    
    with torch.no_grad():
        for imgs, img_paths_batch in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)
            _, z = model(imgs)
            embeddings.append(z.cpu().numpy())
            paths.extend(img_paths_batch)
    
    embeddings = np.vstack(embeddings)
    return embeddings, paths

def cluster_with_kmeans(embeddings, n_clusters=15):
    """Cluster embeddings using K-means"""
    print(f"Clustering into {n_clusters} classes...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels

# ============================================
# STEP 3: Per-Class Anomaly Detection
# ============================================

class PerClassKNNDetector:
    """Per-class anomaly detection using K-nearest neighbors"""
    def __init__(self, k=10):  # Increased k
        self.k = k
        self.class_models = {}
    
    def fit(self, class_id, embeddings):
        """Fit KNN model for one class"""
        print(f"Fitting class {class_id} with {len(embeddings)} samples...")
        knn = NearestNeighbors(n_neighbors=min(self.k, len(embeddings)))
        knn.fit(embeddings)
        self.class_models[class_id] = {
            'knn': knn,
            'embeddings': embeddings
        }
    
    def predict(self, class_id, embeddings):
        """Predict anomaly scores using average distance to k-nearest neighbors"""
        if class_id not in self.class_models:
            # If class not seen in training, return high anomaly score
            return np.ones(len(embeddings)) * 999
        
        knn = self.class_models[class_id]['knn']
        distances, _ = knn.kneighbors(embeddings)
        
        # Average distance to k nearest neighbors
        scores = np.mean(distances, axis=1)
        return scores

class PerClassAEDetector:
    """Per-class anomaly detection using reconstruction error"""
    def __init__(self, model, img_size=128):
        self.model = model
        self.img_size = img_size
        self.class_thresholds = {}
    
    def compute_reconstruction_error(self, img_paths):
        """Compute reconstruction error for images"""
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        
        errors = []
        with torch.no_grad():
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                recon, _ = self.model(img_tensor)
                error = F.mse_loss(recon, img_tensor, reduction='mean')
                errors.append(error.item())
        
        return np.array(errors)
    
    def fit(self, class_id, img_paths):
        """Compute statistics for one class"""
        print(f"Computing reconstruction stats for class {class_id}...")
        errors = self.compute_reconstruction_error(img_paths)
        
        # Store statistics
        self.class_thresholds[class_id] = {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'max': np.max(errors)
        }
    
    def predict(self, img_paths):
        """Predict anomaly scores using reconstruction error"""
        return self.compute_reconstruction_error(img_paths)

# ============================================
# STEP 4: Assign Test Images to Clusters
# ============================================

def assign_to_clusters(kmeans, test_embeddings):
    """Assign test images to nearest cluster"""
    labels = kmeans.predict(test_embeddings)
    return labels

# ============================================
# MAIN PIPELINE
# ============================================

def main():
    # Configuration
    TRAIN_DIR = './Dataset/train'  # CHANGE THIS
    TEST_DIR = './Dataset/test'    # CHANGE THIS
    IMG_SIZE = 224
    N_CLUSTERS = 20
    
    print("="*60)
    print("MVTec Anomaly Detection Pipeline")
    print("="*60)
    
    # Get all image paths
    train_paths = sorted([str(p) for p in Path(TRAIN_DIR).glob('*.png')])
    test_paths = sorted([str(p) for p in Path(TEST_DIR).glob('*.png')])
    
    print(f"\nFound {len(train_paths)} training images")
    print(f"Found {len(test_paths)} test images")
    
    # ============================================
    # PHASE 1: Train Autoencoder
    # ============================================
    print("\n" + "="*60)
    print("PHASE 1: Training Autoencoder")
    print("="*60)
    
    ae_model = train_autoencoder(train_paths, epochs=50, img_size=IMG_SIZE)
    
    # ============================================
    # PHASE 2: Extract Embeddings and Cluster
    # ============================================
    print("\n" + "="*60)
    print("PHASE 2: Extracting Embeddings and Clustering")
    print("="*60)
    
    # Check if embeddings and clusters are already saved
    if os.path.exists('train_embeddings.npy') and os.path.exists('kmeans_model.pkl'):
        print("Loading saved embeddings and clusters...")
        train_embeddings = np.load('train_embeddings.npy')
        train_paths_ordered = np.load('train_paths.npy', allow_pickle=True).tolist()
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        train_labels = kmeans.labels_
        print(f"Loaded embeddings shape: {train_embeddings.shape}")
    else:
        train_embeddings, train_paths_ordered = extract_embeddings(ae_model, train_paths, img_size=IMG_SIZE)
        print(f"Extracted embeddings shape: {train_embeddings.shape}")
        
        kmeans, train_labels = cluster_with_kmeans(train_embeddings, n_clusters=N_CLUSTERS)
        
        # Save for future runs
        np.save('train_embeddings.npy', train_embeddings)
        np.save('train_paths.npy', np.array(train_paths_ordered))
        with open('kmeans_model.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
        print("Saved embeddings and clusters")
    
    # Show cluster distribution
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f"Cluster distribution: {dict(zip(unique, counts))}")
    
    # ============================================
    # PHASE 3: Train Per-Class Anomaly Detectors
    # ============================================
    print("\n" + "="*60)
    print("PHASE 3: Training Per-Class Anomaly Detectors")
    print("="*60)
    
    # Check if detectors are already saved
    if os.path.exists('knn_detector.pkl') and os.path.exists('ae_detector.pkl'):
        print("Loading saved detectors...")
        with open('knn_detector.pkl', 'rb') as f:
            knn_detector = pickle.load(f)
        with open('ae_detector.pkl', 'rb') as f:
            ae_detector = pickle.load(f)
        print("Detectors loaded")
    else:
        # Method 1: KNN-based detector
        knn_detector = PerClassKNNDetector(k=10)
        
        # Method 2: AE reconstruction detector
        ae_detector = PerClassAEDetector(ae_model, img_size=IMG_SIZE)
        
        # Group training data by cluster
        train_by_cluster = {}
        for path, label, emb in zip(train_paths_ordered, train_labels, train_embeddings):
            if label not in train_by_cluster:
                train_by_cluster[label] = {'paths': [], 'embeddings': []}
            train_by_cluster[label]['paths'].append(path)
            train_by_cluster[label]['embeddings'].append(emb)
        
        # Fit detectors for each cluster
        for class_id in range(N_CLUSTERS):
            if class_id in train_by_cluster:
                embeddings = np.array(train_by_cluster[class_id]['embeddings'])
                paths = train_by_cluster[class_id]['paths']
                
                knn_detector.fit(class_id, embeddings)
                ae_detector.fit(class_id, paths)
        
        # Save detectors
        with open('knn_detector.pkl', 'wb') as f:
            pickle.dump(knn_detector, f)
        with open('ae_detector.pkl', 'wb') as f:
            pickle.dump(ae_detector, f)
        print("Saved detectors")
    
    # ============================================
    # PHASE 4: Predict on Test Set
    # ============================================
    print("\n" + "="*60)
    print("PHASE 4: Predicting on Test Set")
    print("="*60)
    
    # Extract test embeddings
    test_embeddings, test_paths_ordered = extract_embeddings(ae_model, test_paths, img_size=IMG_SIZE)
    
    # Assign test images to clusters
    test_labels = assign_to_clusters(kmeans, test_embeddings)
    print(f"Test cluster distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
    
    # Group test data by cluster
    test_by_cluster = {}
    for path, label, emb in zip(test_paths_ordered, test_labels, test_embeddings):
        if label not in test_by_cluster:
            test_by_cluster[label] = {'paths': [], 'embeddings': []}
        test_by_cluster[label]['paths'].append(path)
        test_by_cluster[label]['embeddings'].append(emb)
    
    # Predict anomaly scores - COLLECT ALL FIRST, THEN NORMALIZE GLOBALLY
    all_knn_scores = []
    all_ae_scores = []
    all_paths = []
    
    for class_id in range(N_CLUSTERS):
        if class_id in test_by_cluster:
            paths = test_by_cluster[class_id]['paths']
            embeddings = np.array(test_by_cluster[class_id]['embeddings'])
            
            # Get scores from both methods
            knn_scores = knn_detector.predict(class_id, embeddings)
            ae_scores = ae_detector.predict(paths)
            
            all_knn_scores.extend(knn_scores)
            all_ae_scores.extend(ae_scores)
            all_paths.extend(paths)
    
    # Convert to numpy arrays
    all_knn_scores = np.array(all_knn_scores)
    all_ae_scores = np.array(all_ae_scores)
    
    # Normalize globally (not per-cluster!)
    knn_scores_norm = (all_knn_scores - all_knn_scores.min()) / (all_knn_scores.max() - all_knn_scores.min() + 1e-8)
    ae_scores_norm = (all_ae_scores - all_ae_scores.min()) / (all_ae_scores.max() - all_ae_scores.min() + 1e-8)
    
    # Ensemble: average of both methods
    combined_scores = 0.5 * knn_scores_norm + 0.5 * ae_scores_norm
    
    # Create dictionary
    all_scores = {path: score for path, score in zip(all_paths, combined_scores)}
    
    # ============================================
    # PHASE 5: Generate Submission
    # ============================================
    print("\n" + "="*60)
    print("PHASE 5: Generating Submission")
    print("="*60)
    
    results = []
    for img_path in test_paths:
        img_name = Path(img_path).name
        img_id = int(img_name.replace('.png', ''))
        score = all_scores.get(img_path, 0.5)  # Default to 0.5 if missing
        results.append({'id': img_id, 'prediction': score})
    
    df = pd.DataFrame(results).sort_values('id')
    df.to_csv('submission.csv', index=False)
    
    print(f"\n✓ Submission saved to submission.csv")
    print(f"✓ Total predictions: {len(df)}")
    print(f"✓ Score range: [{df['prediction'].min():.4f}, {df['prediction'].max():.4f}]")
    print(f"✓ Mean score: {df['prediction'].mean():.4f}")
    
    # Show some statistics
    print("\nScore distribution:")
    print(df['prediction'].describe())

if __name__ == '__main__':
    main()