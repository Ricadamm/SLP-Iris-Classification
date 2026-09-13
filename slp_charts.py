import openpyxl
import matplotlib.pyplot as plt

wb = openpyxl.load_workbook('SetosaVersicolor(2).xlsx', data_only=True)

def get_metrics(sheet, start_row):
    ws = wb[sheet]
    epochs, correct, total, sse = [], 0, 0, 0.0
    for r in range(start_row, ws.max_row + 1):
        if ws.cell(r, 1).value and 'epoch' in str(ws.cell(r, 1).value).lower():
            if total > 0:
                epochs.append((correct/total*100, sse/total))
            correct, total, sse = 0, 0, 0.0
        t, p, s = ws.cell(r,7).value, ws.cell(r,16).value, ws.cell(r,18).value
        if t is not None and p is not None and s is not None:
            total += 1
            correct += int(p) == int(t)
            sse += float(s)
    if total > 0:
        epochs.append((correct/total*100, sse/total))
    return epochs

train = get_metrics('Training Data', 5)
val = get_metrics('Validation Data', 4)
x = list(range(1, len(train)+1))

for i, (title, yi, ylabel) in enumerate([
    ('Accuracy Chart', 0, 'Accuracy (%)'),
    ('Loss Chart', 1, 'Loss (Mean SSE)')
]):
    plt.figure(figsize=(8,5))
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
