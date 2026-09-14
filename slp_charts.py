import numpy as np
import matplotlib.pyplot as plt
import openpyxl

# Load data from the Excel "Data" sheet
wb = openpyxl.load_workbook('SetosaVersicolor_AdamRizky.xlsx', data_only=True)
ws = wb['Data']
X = np.array([[ws.cell(r, c).value for c in range(1, 5)] for r in range(1, 101)])
y = np.array([0.0]*50 + [1.0]*50)   # rows 1-50 = Setosa, 51-100 = Versicolor

# Split: first 40 of each class = train, last 10 = validation
train_x = np.vstack([X[:40], X[50:90]])
train_y = np.concatenate([y[:40], y[50:90]])
val_x = np.vstack([X[40:50], X[90:100]])
val_y = np.concatenate([y[40:50], y[90:100]])

# SLP settings
lr = 0.1
bias = 0.5
w = np.full(4, 0.5)
train, val = [], []

for epoch in range(5):
    correct, sse = 0, 0.0
    for x, t in zip(train_x, train_y):
        g = 1 / (1 + np.exp(-(bias + w @ x)))
        err = g - t
        sse += err ** 2
        correct += (round(g) == t)
        grad = 2 * err * g * (1 - g)
        bias -= lr * grad
        w -= lr * grad * x
    train.append((correct / 80 * 100, sse / 80))

    vg = np.array([1 / (1 + np.exp(-(bias + w @ x))) for x in val_x])
    val.append((np.mean(np.round(vg) == val_y) * 100, np.mean((vg - val_y) ** 2)))

    print(f"Epoch {epoch+1}: Train Acc={train[-1][0]:.1f}% Loss={train[-1][1]:.4f} | Val Acc={val[-1][0]:.1f}% Loss={val[-1][1]:.4f}")

# Plot
x = list(range(1, 6))
for i, (title, yi, ylabel) in enumerate([
    ('Accuracy Chart', 0, 'Accuracy (%)'),
    ('Loss Chart', 1, 'Loss (Mean SSE)')
]):
    plt.figure(figsize=(8, 5))
    plt.plot(x, [e[yi] for e in train], 'b-o', label='Training')
    plt.plot(x, [e[yi] for e in val], 'r-o', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title + ' - Training vs Validation')
    plt.legend()
    plt.grid(True)
    plt.xticks(x)
    if yi == 0: plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(['accuracy_chart.png', 'loss_chart.png'][i], dpi=150)

plt.show()
