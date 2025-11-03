import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# =====================================================
# DEVICE
# =====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================================================
# DATASET
# =====================================================
class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        self.image_paths = sorted(list(self.root_dir.glob('*.png')))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return (image, str(img_path.stem)) if not self.is_train else (image, 0)

# =====================================================
# MODEL: Convolutional Autoencoder (with TRUE Bottleneck)
# =====================================================
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        # Based on 5 conv layers with stride 2, for a 256x256 image:
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.final_conv_size = 8
        self.final_conv_channels = 256 # Channels before bottleneck

        # --- Encoder ---
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),       # 256 -> 128
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),      # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),     # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),    # 32 -> 16
            # 🔻 REMOVED last layer that used latent_dim 🔻
            nn.Conv2d(256, self.final_conv_channels, 4, 2, 1), nn.BatchNorm2d(self.final_conv_channels), nn.LeakyReLU(0.2) # 16 -> 8
        )
        
        # --- True Bottleneck ---
        # Flatten the [B, C, 8, 8] feature map and force it through a tiny vector
        self.encoder_fc = nn.Linear(self.final_conv_channels * self.final_conv_size * self.final_conv_size, self.latent_dim)
        
        # --- Decoder ---
        # Take the tiny vector and expand it back
        self.decoder_fc = nn.Linear(self.latent_dim, self.final_conv_channels * self.final_conv_size * self.final_conv_size)
        
        # 🔻 MODIFIED first layer to accept correct channels 🔻
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(self.final_conv_channels, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(), # 8 -> 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), # 16 -> 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),  # 32 -> 64
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),   # 64 -> 128
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),                     # 128 -> 256
        )

    def forward(self, x):
        # Encode
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1) # Flatten: [B, 256*8*8]
        f = self.encoder_fc(x)    # Bottleneck: [B, 32]
        
        # Decode
        x = self.decoder_fc(f)
        x = x.view(x.size(0), self.final_conv_channels, self.final_conv_size, self.final_conv_size) # Un-flatten
        recon = self.decoder_conv(x)
        
        # We return 'f' (the latent vector) just in case, but we won't use it
        return recon, f

# =====================================================
# SSIM LOSS
# =====================================================
def ssim_loss(img1, img2, window_size=11, size_average=True):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    window_size = min(window_size, img1.shape[2], img1.shape[3])
    mu1 = nn.functional.avg_pool2d(img1, window_size, 1, padding=window_size//2, count_include_pad=False)
    mu2 = nn.functional.avg_pool2d(img2, window_size, 1, padding=window_size//2, count_include_pad=False)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq = nn.functional.avg_pool2d(img1*img1, window_size, 1, padding=window_size//2, count_include_pad=False) - mu1_sq
    sigma2_sq = nn.functional.avg_pool2d(img2*img2, window_size, 1, padding=window_size//2, count_include_pad=False) - mu2_sq
    sigma12 = nn.functional.avg_pool2d(img1*img2, window_size, 1, padding=window_size//2, count_include_pad=False) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return (1 - ssim_map.mean()) if size_average else (1 - ssim_map.mean(dim=[1,2,3]))

# =====================================================
# TRAINING (with Early Stopping)
# =====================================================
def train_autoencoder(model, train_loader, val_loader, epochs, lr, patience, model_filename):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Starting training... Monitoring validation loss.")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", unit="batch"):
            data = data.to(device)
            noisy_data = data + 0.02 * torch.randn_like(data)
            noisy_data = torch.clamp(noisy_data, 0, 1)
            
            opt.zero_grad()
            recon, _ = model(noisy_data)
            loss = 0.2 * mse(recon, data) + 0.8 * ssim_loss(recon, data)
            loss.backward()
            opt.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)", unit="batch"):
                data = data.to(device)
                # No noise for validation, we want the true reconstruction error
                recon, _ = model(data)
                loss = 0.2 * mse(recon, data) + 0.8 * ssim_loss(recon, data)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_filename)
            print(f"  ✨ New best model saved! Val Loss: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"  Validation loss did not improve. Counter: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished. Loading best model with val_loss: {best_val_loss:.6f}")
    # Load the best performing model
    model.load_state_dict(torch.load(model_filename, map_location=device))
    return model

# =====================================================
# VISUALIZATION
# =====================================================
def visualize_reconstructions(model, test_loader, device, n_images=5, save_path="reconstructions.png"):
    model.eval()
    print(f"\nGenerating {n_images} reconstruction examples...")
    try: data, img_ids = next(iter(test_loader)) 
    except StopIteration: return
    data = data.to(device)
    with torch.no_grad(): recon, _ = model(data)
    data = data.cpu().numpy()
    recon = recon.cpu().numpy()
    plt.figure(figsize=(15, 6))
    plt.suptitle("Original (Top) vs. Reconstructed (Bottom)", fontsize=16)
    n_images = min(n_images, len(data))
    for i in range(n_images):
        ax = plt.subplot(2, n_images, i + 1)
        img_orig = np.transpose(data[i], (1, 2, 0)) 
        plt.imshow(np.clip(img_orig, 0, 1))
        ax.set_title(f"Original: {img_ids[i]}")
        ax.axis('off')
        ax = plt.subplot(2, n_images, i + 1 + n_images)
        img_recon = np.transpose(recon[i], (1, 2, 0))
        plt.imshow(img_recon)
        ax.set_title(f"Recon: {img_ids[i]}")
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Reconstruction plot saved to: {save_path}")

# =====================================================
# ANOMALY SCORING (FIXED)
# =====================================================
def compute_anomaly_scores(model, test_loader):
    model.eval()
    ids, scores = [], []
    with torch.no_grad():
        for x, names in tqdm(test_loader, desc="Scoring"):
            x = x.to(device)
            recon, _ = model(x)
            mse_val = torch.mean((recon - x)**2, dim=[1,2,3])
            ssim_val = ssim_loss(recon, x, size_average=False)
            score_batch = 0.2 * mse_val + 0.8 * ssim_val
            scores.extend(score_batch.cpu().numpy())
            ids.extend(names)
    return ids, scores

# =====================================================
# MAIN
# =====================================================
def main():
    TRAIN_DIR = './Dataset/train'
    TEST_DIR = './Dataset/test'
    IMG_SIZE = 256
    BATCH_SIZE = 16
    EPOCHS = 50       # Max epochs
    LR = 1e-3
    LATENT = 32 
    NUM_WORKERS = min(4, os.cpu_count())
    
    # --- Early Stopping Params ---
    VALIDATION_SPLIT = 0.1 # Use 10% of training data for validation
    PATIENCE = 5           # Stop if val_loss doesn't improve for 5 epochs
    
    # --- Transformations ---
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # --- Create Datasets ---
    train_ds_full = MVTecDataset(TRAIN_DIR, train_tf, is_train=True)
    test_ds = MVTecDataset(TEST_DIR, test_tf, is_train=False)

    # --- Split Training Data ---
    val_size = int(len(train_ds_full) * VALIDATION_SPLIT)
    train_size = len(train_ds_full) - val_size
    train_subset, val_subset = random_split(train_ds_full, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(42)) # For reproducibility

    print(f"Total train images: {len(train_ds_full)}")
    print(f"Splitting into: Train: {len(train_subset)} | Validation: {len(val_subset)}")

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_subset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    model = ConvAutoencoder(LATENT).to(device)
    
    model_filename = f"model_ae_true_latent{LATENT}_epochs{EPOCHS}_best.pth"
    
    if os.path.exists(model_filename):
        print(f"Loading pre-trained AE: {model_filename}")
        model.load_state_dict(torch.load(model_filename, map_location=device))
    else:
        print(f"Training new model with TRUE latent_dim={LATENT}...")
        model = train_autoencoder(model, train_loader, val_loader, 
                                  epochs=EPOCHS, lr=LR, patience=PATIENCE, 
                                  model_filename=model_filename)

    # --- Visualize reconstructions ---
    vis_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    vis_filename = f"reconstructions_true_latent{LATENT}.png"
    visualize_reconstructions(model, vis_loader, device, n_images=5, save_path=vis_filename)

    print("Calculating anomaly scores based on reconstruction error...")
    ids, scores = compute_anomaly_scores(model, test_loader)

    scores = np.array(scores)
    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    sub_filename = f"submission_true_latent{LATENT}.csv"
    sub = pd.DataFrame({'id': [int(x) for x in ids], 'prediction': scores_normalized})
    sub.sort_values('id').to_csv(sub_filename, index=False)
    print(f"Submission saved to {sub_filename}")
    print(f"Score range: [{scores_normalized.min():.4f}, {scores_normalized.max():.4f}]")

if __name__ == "__main__":
    main()

