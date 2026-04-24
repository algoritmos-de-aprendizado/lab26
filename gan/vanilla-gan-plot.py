import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from lab.dataset import mnist
from lab.gan.discriminadora import Discriminadora
from lab.gan.geradora import Geradora
from lab.rn.mlp import MLP, train_mlp

# ── helpers ────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_mnist_data(l: int, N: int, batch_size: int):
    """Load sklearn digits, return train/test splits and a DataLoader."""
    X_np, y_np = mnist(N * 2)
    n_features = X_np.shape[1]          # 64 for sklearn digits (8x8)

    X_t = torch.tensor(X_np)
    y_t = torch.tensor(y_np, dtype=torch.long)

    split = int(len(X_t) * 0.8)
    X_train, X_test = X_t[:split], X_t[split:]
    y_train, y_test = y_t[:split], y_t[split:]

    data = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    return X_train, y_train, X_test, y_test, n_features, data

def train_and_eval_mlp(X, y, X_test, y_test, n_features):
    """Train an MLP classifier and return it with its test accuracy."""
    mlp = train_mlp(X, y, n_features)
    with torch.no_grad():
        preds = torch.argmax(mlp(X_test), dim=1)
    acc = (preds == y_test).float().mean().item()
    return mlp, acc

def init_plot():
    plt.ion()
    fig, (ax_img, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))
    ax_img.set_title('Generated sample')
    ax_img.axis('off')
    ax_loss.set_title('Loss')
    ax_loss.set_xlabel('Epoch')
    plt.tight_layout()
    plt.show()
    return fig, ax_img, ax_loss

def connect_key_handler(fig, on_key):
    fig.canvas.mpl_connect('key_press_event', on_key)

def update_plot(ax_img, ax_loss, sample, l, pred, mlp_digit, prob_pred, losses_D, losses_G, epoch):
    side = int(sample.numel() ** 0.5)
    img = sample.view(side, side).numpy()
    ax_img.clear()
    ax_img.imshow(img, cmap='gray', vmin=-1, vmax=1)
    ax_img.set_title(f'Epoch {epoch} | D={pred:.2f} | MLP={mlp_digit} ({prob_pred:.2f})')
    ax_img.axis('off')

    ax_loss.clear()
    ax_loss.plot(losses_D, label='D loss')
    ax_loss.plot(losses_G, label='G loss')
    ax_loss.set_title('Loss (sliding window)')
    ax_loss.set_xlabel('Epoch')
    ax_loss.legend()

    plt.pause(0.00001)

def close_plot():
    plt.ioff()
    plt.show()

# ── main ───────────────────────────────────────────────────────────────────────

window_size = 50

set_seeds(42)

l = 8   # sklearn digits are 8x8
N = 4000
batch_size = 200
X, y, X_teste, y_teste, n_features, data = get_mnist_data(l, N, batch_size)
G, D = Geradora(n_features), Discriminadora(n_features)
loss = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0004)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

mlp, acc_teste = train_and_eval_mlp(X, y, X_teste, y_teste, n_features)
print(f'Acurácia de teste da MLP: {acc_teste:.4f}')

fig, ax_img, ax_loss = init_plot()
losses_D = []
losses_G = []

paused = [False]

def on_key(event):
    if event.key == ' ':
        paused[0] = not paused[0]
        print('Pausado' if paused[0] else 'Retomado')
    elif event.key in ['escape', 'q']:
        print('Encerrando...')
        plt.close(fig)

connect_key_handler(fig, on_key)

for epoch in range(100000):
    if not plt.fignum_exists(fig.number):
        break
    for (x,) in data:
        b = x.size(0)
        uns = torch.ones(b, 1)
        zeros = torch.zeros(b, 1)
        z = torch.randn(b, 100)
        x_fake = G(z).detach()
        D_loss = (loss(D(x), uns) + loss(D(x_fake), zeros)) / 2
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()
        z = torch.randn(b, 100)
        G_loss = loss(D(G(z)), uns)
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

    # Sliding window para losses
    losses_D.append(D_loss.item())
    if len(losses_D) > window_size:
        losses_D = losses_D[-window_size:]
    losses_G.append(G_loss.item())
    if len(losses_G) > window_size:
        losses_G = losses_G[-window_size:]

    z = torch.randn(1, 100)
    sample = G(z).detach()
    pred = D(sample).item()
    logits = mlp(sample)
    pred_digit = torch.argmax(logits, dim=1).item()
    mlp_digit = str(pred_digit)
    prob_pred = torch.softmax(logits, dim=1)[0, pred_digit].item()
    update_plot(ax_img, ax_loss, sample, l, pred, mlp_digit, prob_pred, losses_D, losses_G, epoch)
    while paused[0]:
        if not plt.fignum_exists(fig.number):
            break
        plt.pause(0.1)
    if not plt.fignum_exists(fig.number):
        break

close_plot()
