"""
=============================================================
  LAB AM — Perceptron, Separabilidade, MLP e SVM com sklearn
  Disciplina: Aprendizado de Máquina — Graduação em Computação
=============================================================

Objetivos:
  1. Compreender o Perceptron e seus limites (separabilidade linear)
  2. Visualizar a fronteira de decisão aprendida
  3. Resolver problemas não-linearmente separáveis com MLP
  4. Comparar com SVM (kernel linear vs. RBF)

Execute:  python lab_am_redes_neurais.py
Depende:  numpy, matplotlib, scikit-learn  (pip install scikit-learn matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────
# Semente para reprodutibilidade
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────
# Utilitário: plota fronteira de decisão
# ─────────────────────────────────────────────
def plot_fronteira(ax, modelo, X, y, titulo, scaler=None):
    h = 0.03
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grade = np.c_[xx.ravel(), yy.ravel()]
    if scaler:
        grade = scaler.transform(grade)
    Z = modelo.predict(grade).reshape(xx.shape)

    cmap_fundo = ListedColormap(["#FFDDC1", "#C1E1FF"])
    cmap_pontos = ListedColormap(["#E84545", "#2D6A9F"])

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_fundo)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pontos,
               edgecolors="k", s=50, linewidths=0.6)
    acc = accuracy_score(y, modelo.predict(scaler.transform(X) if scaler else X))
    ax.set_title(f"{titulo}\nAcurácia: {acc:.2%}", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

# ═══════════════════════════════════════════════════════════════
# SEÇÃO 1 — PERCEPTRON em dados LINEARMENTE SEPARÁVEIS
# ═══════════════════════════════════════════════════════════════
print("=" * 62)
print("  SEÇÃO 1 — Perceptron em dados linearmente separáveis")
print("=" * 62)

# Gera dataset 2D com boa separabilidade
X_lin, y_lin = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1,
    class_sep=2.0, random_state=SEED
)

perc = Perceptron(max_iter=1000, random_state=SEED)
perc.fit(X_lin, y_lin)

acc_lin = accuracy_score(y_lin, perc.predict(X_lin))
print(f"\nPerceptron — Dataset linear")
print(f"  Épocas até convergência : {perc.n_iter_}")
print(f"  Acurácia (treino)        : {acc_lin:.2%}")
print(f"  Pesos aprendidos         : w={perc.coef_[0]}, b={perc.intercept_[0]:.4f}")

# ═══════════════════════════════════════════════════════════════
# SEÇÃO 2 — PERCEPTRON em XOR (NÃO SEPARÁVEL)
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 62)
print("  SEÇÃO 2 — Perceptron no problema XOR (não-linear)")
print("=" * 62)

# XOR clássico com ruído
X_xor = np.array([[0,0],[0,1],[1,0],[1,1],
                   [0.1,0.1],[0.9,0.9],[0.1,0.9],[0.9,0.1]])
y_xor = np.array([0, 1, 1, 0,  0, 0, 1, 1])

perc_xor = Perceptron(max_iter=1000, random_state=SEED)
perc_xor.fit(X_xor, y_xor)
acc_xor = accuracy_score(y_xor, perc_xor.predict(X_xor))

print(f"\nPerceptron — XOR")
print(f"  Acurácia (treino) : {acc_xor:.2%}")
print("  ⚠  Perceptron não consegue resolver XOR (problema não-linearmente separável).")

# ═══════════════════════════════════════════════════════════════
# SEÇÃO 3 — MLP resolve XOR e dados em Lua (moons)
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 62)
print("  SEÇÃO 3 — MLP (Multi-Layer Perceptron)")
print("=" * 62)

# 3a. XOR com MLP
mlp_xor = MLPClassifier(hidden_layer_sizes=(4,), activation="relu",
                        max_iter=2000, random_state=SEED)
mlp_xor.fit(X_xor, y_xor)
print(f"\nMLP — XOR  |  Acurácia: {accuracy_score(y_xor, mlp_xor.predict(X_xor)):.2%}")

# 3b. Moons com MLP
X_moon, y_moon = make_moons(n_samples=300, noise=0.2, random_state=SEED)
scaler_moon = StandardScaler()
X_moon_s = scaler_moon.fit_transform(X_moon)

mlp_moon = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                         max_iter=2000, random_state=SEED)
mlp_moon.fit(X_moon_s, y_moon)
print(f"MLP — Moons (2 camadas ocultas: 16→8 neurônios)")
print(f"  Épocas    : {mlp_moon.n_iter_}")
print(f"  Loss final: {mlp_moon.loss_:.4f}")
print(f"  Acurácia  : {accuracy_score(y_moon, mlp_moon.predict(X_moon_s)):.2%}")

# 3c. Circles com MLP
X_circ, y_circ = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=SEED)
scaler_circ = StandardScaler()
X_circ_s = scaler_circ.fit_transform(X_circ)

mlp_circ = MLPClassifier(hidden_layer_sizes=(32, 16), activation="tanh",
                         max_iter=3000, random_state=SEED)
mlp_circ.fit(X_circ_s, y_circ)
print(f"\nMLP — Circles (2 camadas ocultas: 32→16 neurônios, tanh)")
print(f"  Acurácia : {accuracy_score(y_circ, mlp_circ.predict(X_circ_s)):.2%}")

# ═══════════════════════════════════════════════════════════════
# SEÇÃO 4 — SVM: kernel linear vs. RBF
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 62)
print("  SEÇÃO 4 — SVM (Support Vector Machine)")
print("=" * 62)

svm_lin   = SVC(kernel="linear", C=1.0, random_state=SEED)
svm_rbf   = SVC(kernel="rbf",    C=1.0, gamma="scale", random_state=SEED)
svm_poly  = SVC(kernel="poly",   degree=3, C=1.0, random_state=SEED)

for nome, modelo in [("SVM Linear", svm_lin),
                     ("SVM RBF",    svm_rbf),
                     ("SVM Poly",   svm_poly)]:
    modelo.fit(X_moon_s, y_moon)
    acc = accuracy_score(y_moon, modelo.predict(X_moon_s))
    svs = modelo.n_support_.sum()
    print(f"\n{nome}  —  Moons")
    print(f"  Acurácia          : {acc:.2%}")
    print(f"  Vetores de suporte: {svs}")

# ═══════════════════════════════════════════════════════════════
# SEÇÃO 5 — RELATÓRIO COMPARATIVO completo em Moons + Circles
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 62)
print("  SEÇÃO 5 — Relatório comparativo (Moons)")
print("=" * 62)

modelos_comp = {
    "Perceptron"    : Perceptron(max_iter=1000, random_state=SEED),
    "MLP (16-8)"    : MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=SEED),
    "SVM Linear"    : SVC(kernel="linear", C=1),
    "SVM RBF"       : SVC(kernel="rbf", C=1, gamma="scale"),
}

for nome, modelo in modelos_comp.items():
    modelo.fit(X_moon_s, y_moon)
    pred = modelo.predict(X_moon_s)
    print(f"\n{nome}")
    print(classification_report(y_moon, pred, target_names=["Classe 0", "Classe 1"],
                                 zero_division=0))

# ═══════════════════════════════════════════════════════════════
# VISUALIZAÇÕES — Janelas separadas por assunto
# ═══════════════════════════════════════════════════════════════
fig_fronteiras, axes_fronteiras = plt.subplots(2, 4, figsize=(18, 10))
fig_fronteiras.suptitle(
    "Lab AM — Perceptron · Separabilidade · MLP · SVM\n"
    "Fronteiras de Decisão Aprendidas",
    fontsize=14, fontweight="bold", y=0.98
)

# ── Linha 1: Perceptron (linear), Perceptron (XOR), MLP Moon, MLP Circle
ax1 = axes_fronteiras[0, 0]
plot_fronteira(ax1, perc, X_lin, y_lin, "Perceptron\n(Linear — separável)")

ax2 = axes_fronteiras[0, 1]
plot_fronteira(ax2, perc_xor, X_xor, y_xor, "Perceptron\n(XOR — falha esperada)")

ax3 = axes_fronteiras[0, 2]
plot_fronteira(ax3, mlp_moon, X_moon, y_moon, "MLP 16→8\n(Moons)", scaler=scaler_moon)

ax4 = axes_fronteiras[0, 3]
plot_fronteira(ax4, mlp_circ, X_circ, y_circ, "MLP 32→16 tanh\n(Circles)", scaler=scaler_circ)

# ── Linha 2: SVM Linear, SVM RBF, SVM Poly em Moons + SVM RBF em Circles
ax5 = axes_fronteiras[1, 0]
svm_lin_new = SVC(kernel="linear", C=1)
svm_lin_new.fit(X_moon_s, y_moon)
plot_fronteira(ax5, svm_lin_new, X_moon, y_moon, "SVM Kernel Linear\n(Moons)", scaler=scaler_moon)

ax6 = axes_fronteiras[1, 1]
svm_rbf_new = SVC(kernel="rbf", C=1, gamma="scale")
svm_rbf_new.fit(X_moon_s, y_moon)
plot_fronteira(ax6, svm_rbf_new, X_moon, y_moon, "SVM Kernel RBF\n(Moons)", scaler=scaler_moon)

ax7 = axes_fronteiras[1, 2]
svm_poly_new = SVC(kernel="poly", degree=3, C=1)
svm_poly_new.fit(X_moon_s, y_moon)
plot_fronteira(ax7, svm_poly_new, X_moon, y_moon, "SVM Kernel Polinomial\n(Moons)", scaler=scaler_moon)

ax8 = axes_fronteiras[1, 3]
svm_rbf_circ = SVC(kernel="rbf", C=1, gamma="scale")
svm_rbf_circ.fit(X_circ_s, y_circ)
plot_fronteira(ax8, svm_rbf_circ, X_circ, y_circ, "SVM Kernel RBF\n(Circles)", scaler=scaler_circ)

fig_fronteiras.tight_layout(rect=(0, 0, 1, 0.96))

# ─────────────────────────────────────────────
# CURVA DE PERDA — MLP Moon
# ─────────────────────────────────────────────
fig_perdas, axes_perdas = plt.subplots(1, 2, figsize=(12, 4))
fig_perdas.suptitle("Curvas de Perda — MLP", fontsize=13, fontweight="bold")

ax9, ax10 = axes_perdas

ax9.plot(mlp_moon.loss_curve_, color="#2D6A9F", lw=1.8)
ax9.set_title("Curva de Perda — MLP 16→8 relu (Moons)")
ax9.set_xlabel("Época"); ax9.set_ylabel("Log-Loss")
ax9.grid(alpha=0.3)

ax10.plot(mlp_circ.loss_curve_, color="#E84545", lw=1.8)
ax10.set_title("Curva de Perda — MLP 32→16 tanh (Circles)")
ax10.set_xlabel("Época"); ax10.set_ylabel("Log-Loss")
ax10.grid(alpha=0.3)

fig_perdas.tight_layout(rect=(0, 0, 1, 0.92))
print("\n[✓] Exibindo as visualizações em janelas separadas por assunto.")
plt.show()

# ═══════════════════════════════════════════════════════════════
# RESUMO CONCEITUAL IMPRESSO NO TERMINAL
# ═══════════════════════════════════════════════════════════════
print("=" * 62)
resumo = """
PERCEPTRON
  • Neurônio único com saída binária via função degrau.
  • Converge apenas para dados linearmente separáveis.
  • XOR não é linearmente separável → Perceptron falha.

SEPARABILIDADE LINEAR
  • Um dataset é L.S. se existe um hiperplano que separa as classes.
  • Moons e Circles NÃO são linearmente separáveis.

MLP (Multi-Layer Perceptron)
  • Múltiplas camadas ocultas com funções de ativação não-lineares
    (ReLU, Tanh, Sigmoid).
  • Aprende fronteiras arbitrariamente complexas via backpropagation.
  • Hiperparâmetros importantes: arquitetura, lr, ativação, regularização.

SVM (Support Vector Machine)
  • Maximiza a margem entre classes.
  • Kernel Linear: separação linear no espaço original.
  • Kernel RBF   : mapeia para espaço de alta dimensão onde seja linearmente separável.
  • Kernel Poly  : fronteira polinomial de grau d.
  • C controla compromisso entre margem e classificações erradas.
"""
print(resumo)
print("=" * 62)
