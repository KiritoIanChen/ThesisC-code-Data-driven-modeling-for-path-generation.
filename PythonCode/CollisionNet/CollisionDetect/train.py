import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from GJKNN import CollisionNet
# 1. 加载数据
mat = scipy.io.loadmat("OBB_GJK_Data_49xN.mat")
datas = mat["datas"]

np.random.shuffle(datas)
X = datas[:, :48]
y = datas[:, 48].astype(int)

# 2. 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=15000, shuffle=False)

# 3. 归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 转tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# # 5. 定义改进网络结构
# class ImprovedNet(nn.Module):
#     def __init__(self):
#         super(ImprovedNet, self).__init__()
#         self.fc1 = nn.Linear(48, 64)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.fc2 = nn.Linear(64, 32)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.dropout = nn.Dropout(0.3)
#         self.fc3 = nn.Linear(32, 16)
#         self.out = nn.Linear(16, 2)
#
#     def forward(self, x):
#         x = torch.relu(self.bn1(self.fc1(x)))
#         x = torch.relu(self.bn2(self.dropout(self.fc2(x))))
#         x = torch.relu(self.fc3(x))
#         x = self.out(x)  # 交叉熵loss包含softmax
#         return x

model = CollisionNet()

# 6. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 7. 训练参数
batch_size = 128
epochs = 50
dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_losses = []
train_accuracies = []

# 8. 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        _, predicted = torch.max(preds, 1)
        correct += (predicted == yb).sum().item()
        total += yb.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# 9. 训练曲线绘制
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.legend()
plt.savefig("training_loss.png")

plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), train_accuracies, label="Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.grid()
plt.legend()
plt.savefig("test_accuracy.png")

plt.tight_layout()
plt.show()

# 10. 测试评估和混淆矩阵
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, preds = torch.max(outputs, 1)

acc = accuracy_score(y_test, preds.numpy())
print(f"Test Accuracy: {acc:.4f}")

cm = confusion_matrix(y_test, preds.numpy())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Collision", "Collision"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix on Test Set")
plt.show()
plt.savefig("confusion_matrix.png")

# 11. 保存模型
torch.save(model, "GJKNet.pt")  # 包括模型结构和权重
joblib.dump(scaler, "gjk_scaler.save")  # 保存归一化器