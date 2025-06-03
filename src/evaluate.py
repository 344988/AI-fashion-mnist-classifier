import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os

from model import NeuralNet  # Импорт модели

# === Подготовка данных ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# === Загрузка модели ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# === Оценка точности и сбор предсказаний ===
all_preds = []
all_labels = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# === Матрица ошибок ===
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
os.makedirs("results", exist_ok=True)
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.close()

# === Примеры ошибок ===
misclassified = [(img, pred, true) for (img, pred, true) in zip(test_dataset.data, all_preds, all_labels) if pred != true]

plt.figure(figsize=(10, 6))
for i, (img, pred, true) in enumerate(misclassified[:8]):
    plt.subplot(2, 4, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Предсказано: {pred}\nИстинно: {true}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('results/misclassified_examples.png')
plt.close()
print("Сохранены результаты в папку 'results/'")
