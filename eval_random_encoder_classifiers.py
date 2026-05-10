import torch
import torch.nn as nn
import numpy as np


# ── WavShape encoder architecture (must match classifier_random_encoder.py) ──
class DenseEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        prev_size = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_size, hidden_dim))
            layers.append(nn.ReLU())
            prev_size = hidden_dim
        layers.append(nn.Linear(prev_size, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x.float())


def apply_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


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


# ── Rebuild the same frozen random encoder (same seed → same weights) ──
torch.manual_seed(42)
random_encoder = DenseEncoder(in_dim=512, hidden_dims=[256, 128], out_dim=64)
random_encoder.apply(apply_xavier)
random_encoder.eval()

# ── Load test data (subset1) and pass through frozen encoder ────
original_embeddings_test = torch.from_numpy(
    np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_original_embeddings.npy")
).float()
y_pub_test  = torch.from_numpy(
    np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_public_label.npy")
).long()
y_priv_test = torch.from_numpy(
    np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_private_label.npy")
).long()

with torch.no_grad():
    X_rand_test = random_encoder(original_embeddings_test)

# ── Load classifiers from saved checkpoints ──────────────────────
random_public_classifier  = Classifier(64, 32, 2)
random_private_classifier = Classifier(64, 32, 2)

random_public_classifier.load_state_dict(
    torch.load("random_public_classifier_950.pt", map_location="cpu")
)
random_private_classifier.load_state_dict(
    torch.load("random_private_classifier_950.pt", map_location="cpu")
)

# ── Evaluate ─────────────────────────────────────────────────────
acc_rand_pub  = accuracy(random_public_classifier,  X_rand_test, y_pub_test)
acc_rand_priv = accuracy(random_private_classifier, X_rand_test, y_priv_test)

majority_pub  = max(y_pub_test.float().mean().item(),  1 - y_pub_test.float().mean().item())
majority_priv = max(y_priv_test.float().mean().item(), 1 - y_priv_test.float().mean().item())

print("=" * 55)
print("TEST ACCURACY — RANDOM ENCODER (Xavier init, frozen)")
print("=" * 55)
print(f"  Random Encoder → public  (wall):    {acc_rand_pub:.3f}")
print(f"  Random Encoder → private (bedroom): {acc_rand_priv:.3f}")
print()
print(f"  Majority baseline — public:          {majority_pub:.3f}")
print(f"  Majority baseline — private:         {majority_priv:.3f}")
print("=" * 55)
