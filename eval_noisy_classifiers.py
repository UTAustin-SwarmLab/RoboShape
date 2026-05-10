import torch
import torch.nn as nn
import numpy as np


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


# ── Load test data (clean subset1, no noise) ───────────────────
X_orig_test = torch.from_numpy(
    np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_original_embeddings.npy")
).float()
y_pub_test  = torch.from_numpy(
    np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_public_label.npy")
).long()
y_priv_test = torch.from_numpy(
    np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_private_label.npy")
).long()

# ── Load noisy classifiers from saved weights ──────────────────
noisy_public_classifier  = Classifier(512, 256, 2)
noisy_private_classifier = Classifier(512, 256, 2)

noisy_public_classifier.load_state_dict(
    torch.load("noisy_public_classifier_100.pt", map_location="cpu")
)
noisy_private_classifier.load_state_dict(
    torch.load("noisy_private_classifier_100.pt", map_location="cpu")
)

# ── Evaluate ───────────────────────────────────────────────────
acc_noisy_pub  = accuracy(noisy_public_classifier,  X_orig_test, y_pub_test)
acc_noisy_priv = accuracy(noisy_private_classifier, X_orig_test, y_priv_test)

majority_pub  = max(y_pub_test.float().mean().item(),  1 - y_pub_test.float().mean().item())
majority_priv = max(y_priv_test.float().mean().item(), 1 - y_priv_test.float().mean().item())

print("=" * 55)
print("TEST ACCURACY — NOISY ORIGINAL EMBEDDINGS (σ=1.0)")
print("=" * 55)
print(f"  Noisy Original → public  (wall):    {acc_noisy_pub:.3f}")
print(f"  Noisy Original → private (bedroom): {acc_noisy_priv:.3f}")
print()
print(f"  Majority baseline — public:          {majority_pub:.3f}")
print(f"  Majority baseline — private:         {majority_priv:.3f}")
print("=" * 55)
