# 导入必要的库
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pdb

# 设置设备为GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据集
df_pima = pd.read_csv('hypothyroid.csv')

# 对分类变量进行Label Encoding
for col in df_pima.columns:
    le = LabelEncoder()
    df_pima[col] = le.fit_transform(df_pima[col])

# 特征和标签
feature_cols = ["age","sex","on thyroxine","query on thyroxine","on antithyroid medication","sick","pregnant",
                "thyroid surgery","I131 treatment","query hypothyroid","query hyperthyroid","lithium",
                "goitre","tumor","hypopituitary","psych","TSH measured","TSH","T3 measured","T3",
                "TT4 measured","TT4","T4U measured","T4U","FTI measured","FTI","TBG measured","TBG",
                "referral source"]
X = df_pima[feature_cols].values
y = df_pima["binaryClass"].values

# 数据集划分（80%训练集，20%测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=47)

# 将数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# 创建DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 定义神经网络模型
class DNNModel(nn.Module):
    def __init__(self, input_dim, layers, neurons_per_layer, dropout_rate, activation_fn):
        super(DNNModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(nn.Linear(input_dim, neurons_per_layer[0]))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 隐藏层
        for i in range(1, layers):
            self.layers.append(nn.Linear(neurons_per_layer[i-1], neurons_per_layer[i]))
            self.layers.append(nn.Dropout(dropout_rate))
        
        # 输出层（2分类）
        self.output_layer = nn.Linear(neurons_per_layer[-1], 2)
        self.activation_fn = activation_fn

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        x = F.log_softmax(self.output_layer(x), dim=1)
        return x

# 权重初始化函数
def initialize_weights(model, init_type):
    if init_type == 'xavier':
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    elif init_type == 'he':
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

# 定义训练和评估函数
def train_and_evaluate_model(batch_size, epochs, layers, neurons_per_layer, dropout_rate, learning_rate, activation_fn, init_type, optimizer_type):
    model = DNNModel(X_train.shape[1], layers, neurons_per_layer, dropout_rate, activation_fn).to(device)
    
    # 权重初始化
    initialize_weights(model, init_type)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 选择优化器
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 创建DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 训练循环
    train_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

    # 测试集评估
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted.cpu())
        
        # 计算AUC
        fpr, tpr, _ = roc_curve(y_test, outputs[:, 1].cpu())
        roc_auc = auc(fpr, tpr)
    
    return roc_auc, accuracy, train_losses[-1]

# 网格搜索优化超参数
best_auc = 0.0
best_params = None
best_model_performance = None

# 参数网格
batch_sizes = [16, 32, 64]
epoch_values = [30, 50, 100]
layers_values = [2, 3, 4]
neurons_values = [[128, 64], [256, 128, 64], [512, 256, 128, 64]]
dropout_values = [0.3, 0.4, 0.5, 0.6]
learning_rates = [0.001, 0.0005, 0.0001]
activations = [F.relu, F.leaky_relu, torch.sigmoid, torch.tanh]
initializations = ['xavier', 'he']
optimizers = ['adam', 'sgd']

# 尝试不同参数组合
for batch_size in batch_sizes:
    for epochs in epoch_values:
        for layers, neurons in zip(layers_values, neurons_values):
            for dropout_rate in dropout_values:
                for lr in learning_rates:
                    for activation_fn in activations:
                        for init_type in initializations:
                            for optimizer_type in optimizers:
                                auc_score, accuracy, loss = train_and_evaluate_model(
                                    batch_size, epochs, layers, neurons, dropout_rate, lr, activation_fn, init_type, optimizer_type
                                )
                                print(f"Batch size: {batch_size}, Epochs: {epochs}, Layers: {layers}, Dropout: {dropout_rate}, Learning Rate: {lr}")
                                print(f"Init: {init_type}, Optimizer: {optimizer_type}")
                                print(f"AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
                                
                                if auc_score > best_auc:
                                    best_auc = auc_score
                                    best_params = (batch_size, epochs, layers, neurons, dropout_rate, lr, activation_fn, init_type, optimizer_type)
                                    best_model_performance = (accuracy, loss)

print(f"Best AUC: {best_auc:.4f} with params {best_params}")
print(f"Best model accuracy: {best_model_performance[0]:.4f}, Loss: {best_model_performance[1]:.4f}")
