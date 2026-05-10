import torch
import torch.nn as nn
import numpy as np
import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


torch.manual_seed(42)

# ── Gaussian DP noise scale ────────────────────────────────────
SIGMA = 1.0   # increase for more privacy, decrease for less noise

# ── Load training data (subset2) ──────────────────────────────
original_embeddings_train = np.load("/home/obiwan/mirac/sonata/scannet_subset2_tensors/scannet_subset2_original_embeddings.npy")
public_labels_train       = np.load("/home/obiwan/mirac/sonata/scannet_subset2_tensors/scannet_subset2_public_label.npy")
private_labels_train      = np.load("/home/obiwan/mirac/sonata/scannet_subset2_tensors/scannet_subset2_private_label.npy")

# ── Add Gaussian noise to training embeddings (once, before training) ──
original_train_tensor  = torch.from_numpy(original_embeddings_train).float()
noise                  = torch.randn_like(original_train_tensor) * SIGMA
noisy_embeddings_train = original_train_tensor + noise

y_pub_train  = torch.from_numpy(public_labels_train).long()
y_priv_train = torch.from_numpy(private_labels_train).long()

# ── Load test data (subset1, no noise) ────────────────────────
X_orig_test = torch.from_numpy(
    np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_original_embeddings.npy")
).float()
y_pub_test  = torch.from_numpy(
    np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_public_label.npy")
).long()
y_priv_test = torch.from_numpy(
    np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_private_label.npy")
).long()

# ── Classifiers ────────────────────────────────────────────────
noisy_public_classifier  = Classifier(512, 256, 2)
noisy_private_classifier = Classifier(512, 256, 2)

criterion              = nn.CrossEntropyLoss()
optimizer_noisy_public  = torch.optim.Adam(noisy_public_classifier.parameters(),  lr=0.001)
optimizer_noisy_private = torch.optim.Adam(noisy_private_classifier.parameters(), lr=0.001)

# ── Loss logs ──────────────────────────────────────────────────
epochs = 500
loss_noisy_public       = []
loss_noisy_private      = []
loss_noisy_public_test  = []
loss_noisy_private_test = []

# ── Training loop ──────────────────────────────────────────────
for i in range(epochs):

    y_pred_noisy_public  = noisy_public_classifier(noisy_embeddings_train)
    y_pred_noisy_private = noisy_private_classifier(noisy_embeddings_train)

    l_noisy_pub  = criterion(y_pred_noisy_public,  y_pub_train)
    l_noisy_priv = criterion(y_pred_noisy_private, y_priv_train)

    loss_noisy_public.append(l_noisy_pub.item())
    loss_noisy_private.append(l_noisy_priv.item())

    if i % 10 == 0:
        print(f"Epoch: {i}  Loss Noisy Public: {loss_noisy_public[-1]:.4f}"
              f"  Loss Noisy Private: {loss_noisy_private[-1]:.4f}")

    optimizer_noisy_public.zero_grad()
    optimizer_noisy_private.zero_grad()
    l_noisy_pub.backward()
    l_noisy_priv.backward()
    optimizer_noisy_public.step()
    optimizer_noisy_private.step()

    # Checkpoint every 50 epochs (in train mode, right after optimizer step)
    if i % 50 == 0:
        torch.save(noisy_public_classifier.state_dict(),  f"noisy_public_classifier_{i}.pt")
        torch.save(noisy_private_classifier.state_dict(), f"noisy_private_classifier_{i}.pt")

    # Evaluate on clean test set
    noisy_public_classifier.eval()
    noisy_private_classifier.eval()

    with torch.no_grad():
        logits_noisy_pub  = noisy_public_classifier(X_orig_test)
        logits_noisy_priv = noisy_private_classifier(X_orig_test)

        loss_noisy_public_test.append(criterion(logits_noisy_pub,  y_pub_test).item())
        loss_noisy_private_test.append(criterion(logits_noisy_priv, y_priv_test).item())

    noisy_public_classifier.train()
    noisy_private_classifier.train()


# ── Plots ──────────────────────────────────────────────────────
epoch_range = range(epochs)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epoch_range, loss_noisy_public,      label="Train", color="steelblue")
axes[0].plot(epoch_range, loss_noisy_public_test, label="Test",  color="steelblue", linestyle="--")
axes[0].set_title(f"Noisy Original (σ={SIGMA}) — Public (wall) loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("CrossEntropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epoch_range, loss_noisy_private,      label="Train", color="seagreen")
axes[1].plot(epoch_range, loss_noisy_private_test, label="Test",  color="seagreen", linestyle="--")
axes[1].set_title(f"Noisy Original (σ={SIGMA}) — Private (bedroom) loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("CrossEntropy Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_noisy.png", dpi=150)
plt.close()
print("Saved: loss_noisy.png")
