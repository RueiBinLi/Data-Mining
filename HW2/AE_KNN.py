import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from PIL import Image
from tqdm import tqdm

# ----------------------------
# 1. Dataset
# ----------------------------
class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('L')  # grayscale
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_data = MVTecDataset('./Dataset/train', transform)
test_data = MVTecDataset('./Dataset/test', transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ----------------------------
# 2. Convolutional Autoencoder
# ----------------------------
class ConvAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 16 * 16),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAE(latent_dim=128).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# 3. Train Autoencoder
# ----------------------------
n_epochs = 100
model.train()
for epoch in range(n_epochs):
    total_loss = 0
    for imgs in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        x_hat, _ = model(imgs)
        loss = criterion(x_hat, imgs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {total_loss/len(train_loader):.4f}")

# ----------------------------
# 4. Extract Latent Features
# ----------------------------
def extract_latents(dataloader):
    model.eval()
    latents = []
    recon_losses = []
    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(device)
            x_hat, z = model(imgs)
            latents.append(z.cpu().numpy())
            loss = torch.mean((x_hat - imgs) ** 2, dim=[1, 2, 3])
            recon_losses.append(loss.cpu().numpy())
    return np.concatenate(latents), np.concatenate(recon_losses)

train_latent, _ = extract_latents(train_loader)
test_latent, test_recon = extract_latents(test_loader)

# ----------------------------
# 5. One-Class SVM on Latent
# ----------------------------
scaler_mean = train_latent.mean(axis=0)
scaler_std = train_latent.std(axis=0) + 1e-6
train_latent = (train_latent - scaler_mean) / scaler_std
test_latent = (test_latent - scaler_mean) / scaler_std

svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
svm.fit(train_latent)
svm_scores = -svm.decision_function(test_latent)  # higher means more anomalous

# ----------------------------
# 6. Combine Scores (optional)
# ----------------------------
final_score = 0.7 * svm_scores + 0.3 * test_recon  # weighted combination

# ----------------------------
# 7. Evaluate (if you have labels)
# ----------------------------
# y_true = np.loadtxt("ground_truth.txt")  # 0=normal, 1=anomaly
# auc = roc_auc_score(y_true, final_score)
# print("AUC =", auc)

# ----------------------------
# 8. Kaggle Submission
# ----------------------------
import pandas as pd
submission = pd.DataFrame({
    "id": np.arange(len(final_score)),
    "prediction": (final_score > np.median(final_score)).astype(int)
})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")
