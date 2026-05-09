import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1,):
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
 
original_embeddings_train = np.load("/home/obiwan/mirac/sonata/scannet_subset2_tensors/scannet_subset2_original_embeddings.npy")
roboshape_embeddings_train = np.load("/home/obiwan/mirac/sonata/scannet_subset2_tensors/scannet_subset2_roboshape_embeddings.npy")     
public_lables_train = np.load("/home/obiwan/mirac/sonata/scannet_subset2_tensors/scannet_subset2_public_label.npy")
private_lables_train = np.load("/home/obiwan/mirac/sonata/scannet_subset2_tensors/scannet_subset2_private_label.npy")

original_embeddings_test = np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_original_embeddings.npy")
roboshape_embeddings_test = np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_roboshape_embeddings.npy")     
public_lables_test = np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_public_label.npy")
private_lables_test = np.load("/home/obiwan/mirac/sonata/scannet_subset1_tensors/scannet_subset1_private_label.npy")

X_orig_test  = torch.from_numpy(original_embeddings_test)
X_robo_test  = torch.from_numpy(roboshape_embeddings_test)
y_pub_test   = torch.from_numpy(public_lables_test).long()
y_priv_test  = torch.from_numpy(private_lables_test).long()

# Accuracy lists per epoch
acc_orig_pub, acc_robo_pub = [], []
acc_orig_priv, acc_robo_priv = [], []

original_public_classifier = Classifier(512, 256, 2)   # out_dim=2: one score per class
roboshape_public_classifier = Classifier(64, 32, 2)
original_private_classifier = Classifier(512, 256, 2)
roboshape_private_classifier = Classifier(64, 32, 2)

criterion = nn.CrossEntropyLoss()
optimizer_original_public = torch.optim.Adam(original_public_classifier.parameters(), lr=0.001)
optimizer_roboshape_public = torch.optim.Adam(roboshape_public_classifier.parameters(), lr=0.001)
optimizer_original_private = torch.optim.Adam(original_private_classifier.parameters(), lr=0.001)
optimizer_roboshape_private = torch.optim.Adam(roboshape_private_classifier.parameters(), lr=0.001)

epochs_original = 500
epochs_roboshape = 1000
loss_original_public = []
loss_roboshape_public = []
loss_original_private = []
loss_roboshape_private = []

loss_original_public_test = []
loss_roboshape_public_test = []
loss_original_private_test = []
loss_roboshape_private_test = []

for i in range (epochs_original):
    y_pred_original_public = original_public_classifier.forward(torch.from_numpy(original_embeddings_train))
    y_pred_original_private = original_private_classifier.forward(torch.from_numpy(original_embeddings_train))


    # Keep tensors for .backward(), store float values for logging
    l_orig_pub  = criterion(y_pred_original_public,   torch.from_numpy(public_lables_train).long())
    l_orig_priv = criterion(y_pred_original_private,  torch.from_numpy(private_lables_train).long())

    loss_original_public.append(l_orig_pub.item())
    loss_original_private.append(l_orig_priv.item())

    if i % 10 == 0:
        print("Epoch:", i, "Loss Original Public:", loss_original_public[-1],  "Loss Original Private:", loss_original_private[-1])

    optimizer_original_public.zero_grad()
    optimizer_original_private.zero_grad()

    l_orig_pub.backward()
    l_orig_priv.backward()

    optimizer_original_public.step()
    optimizer_original_private.step()

    # Evaluate on test set
    original_public_classifier.eval()
    original_private_classifier.eval()
    with torch.no_grad():
        logits_orig_pub  = original_public_classifier(X_orig_test)
        logits_orig_priv = original_private_classifier(X_orig_test)
        loss_original_public_test.append(criterion(logits_orig_pub,  torch.from_numpy(public_lables_test).long()).item())
        loss_original_private_test.append(criterion(logits_orig_priv, torch.from_numpy(private_lables_test).long()).item())

    # Back to training mode for next epoch
    original_public_classifier.train()
    original_private_classifier.train()



        
    
for i in range(epochs_roboshape):
    roboshape_public_classifier.train()
    roboshape_private_classifier.train()

    y_pred_roboshape_public  = roboshape_public_classifier(torch.from_numpy(roboshape_embeddings_train))
    y_pred_roboshape_private = roboshape_private_classifier(torch.from_numpy(roboshape_embeddings_train))

    l_robo_pub  = criterion(y_pred_roboshape_public,  torch.from_numpy(public_lables_train).long())
    l_robo_priv = criterion(y_pred_roboshape_private, torch.from_numpy(private_lables_train).long())

    loss_roboshape_public.append(l_robo_pub.item())
    loss_roboshape_private.append(l_robo_priv.item())

    if i % 10 == 0:
        print("Epoch:", i, "Loss Roboshape Public:", loss_roboshape_public[-1], "Loss Roboshape Private:", loss_roboshape_private[-1])

    optimizer_roboshape_public.zero_grad()
    optimizer_roboshape_private.zero_grad()
    l_robo_pub.backward()
    l_robo_priv.backward()
    optimizer_roboshape_public.step()
    optimizer_roboshape_private.step()

    with torch.no_grad():
        logits_robo_pub  = roboshape_public_classifier(X_robo_test)
        logits_robo_priv = roboshape_private_classifier(X_robo_test)

        l_robo_pub_test  = criterion(logits_robo_pub,   torch.from_numpy(public_lables_test).long())
        l_robo_priv_test = criterion(logits_robo_priv,  torch.from_numpy(private_lables_test).long())

        loss_roboshape_public_test.append(l_robo_pub_test.item())
        loss_roboshape_private_test.append(l_robo_priv_test.item())

torch.save(original_public_classifier.state_dict(), "original_public_classifier.pt")
torch.save(roboshape_public_classifier.state_dict(), "roboshape_public_classifier.pt")
torch.save(original_private_classifier.state_dict(), "original_private_classifier.pt")
torch.save(roboshape_private_classifier.state_dict(), "roboshape_private_classifier.pt")

# ── Plots ─────────────────────────────────────────────────────
# x-axis ranges (different epoch counts per loop)
orig_range  = range(epochs_original)
robo_range  = range(epochs_roboshape)

# Plot 1: Public (wall) label — train vs test loss
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(orig_range, loss_original_public,      label="Train", color="steelblue")
axes[0].plot(orig_range, loss_original_public_test, label="Test",  color="steelblue", linestyle="--")
axes[0].set_title("Original embeddings — Public (wall) loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("CrossEntropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(robo_range, loss_roboshape_public,      label="Train", color="darkorange")
axes[1].plot(robo_range, loss_roboshape_public_test, label="Test",  color="darkorange", linestyle="--")
axes[1].set_title("Roboshape embeddings — Public (wall) loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("CrossEntropy Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_public.png", dpi=150)
plt.close()
print("Saved: loss_public.png")

# Plot 2: Private (bedroom) label — train vs test loss
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(orig_range, loss_original_private,      label="Train", color="seagreen")
axes[0].plot(orig_range, loss_original_private_test, label="Test",  color="seagreen", linestyle="--")
axes[0].set_title("Original embeddings — Private (bedroom) loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("CrossEntropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(robo_range, loss_roboshape_private,      label="Train", color="crimson")
axes[1].plot(robo_range, loss_roboshape_private_test, label="Test",  color="crimson", linestyle="--")
axes[1].set_title("Roboshape embeddings — Private (bedroom) loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("CrossEntropy Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_private.png", dpi=150)
plt.close()
print("Saved: loss_private.png")

