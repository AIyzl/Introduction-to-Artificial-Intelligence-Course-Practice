import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 构建模型
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN_LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # CNN层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=input_size,
                               kernel_size=3)  # 输出通道数改为input_size
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        # Dropout层
        self.dropout = nn.Dropout(0.5)  # 添加一个Dropout层，丢弃概率为0.5
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # CNN前向传播
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)

        # 转换CNN输出的形状以适应LSTM
        x = x.permute(0, 2, 1)

        # LSTM前向传播
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        # 只使用LSTM输出序列的最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

def main():
    # 数据预处理
    df = pd.read_csv('Data.csv')
    x = df[df.columns[0: -2]].values
    y = df[df.columns[-1]].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=123)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_train = x_train.unsqueeze(1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    x_test = x_test.unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)



    # 定义超参数
    input_size = 8  # 输入大小，修改为8
    hidden_size = 128  # LSTM隐藏层大小
    num_layers = 2  # LSTM层数
    num_classes = 2  # 分类数
    learning_rate = 0.1
    num_epochs = 1500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型、损失函数和优化器
    model = CNN_LSTM_Model(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    #
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # 模型训练
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # 测试模式
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test.to(device))
            test_loss = criterion(test_outputs, y_test.to(device))
            test_losses.append(test_loss.item())

        # 释放GPU内存
        torch.cuda.empty_cache()
    # 保存模型
    PATH = './model_CNN_LSTM.pt'
    torch.save(model, PATH)
    print('模型保存成功！')

    # 计算训练集和测试集的准确率
    model.eval()
    with torch.no_grad():
        # 训练集准确率
        y_train_pred = model(x_train.to(device)).argmax(dim=1)
        train_accuracy = torch.sum(y_train_pred == y_train.to(device)).item() / len(y_train)

        # 测试集准确率
        y_test_pred = model(x_test.to(device)).argmax(dim=1)
        test_accuracy = torch.sum(y_test_pred == y_test.to(device)).item() / len(y_test)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 绘制训练和测试损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制真实值和预测值的散点图
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train.to(device)).argmax(dim=1)
        plt.scatter(y_train.cpu().numpy(), y_pred.cpu().numpy(), alpha=0.5)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        plt.show()

    # 绘制混淆矩阵图
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test.to(device)).argmax(dim=1)
        cm = confusion_matrix(y_test.cpu().numpy(), y_pred.cpu().numpy())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        # 计算精确率、召回率、F1分数和AUC值
        y_pred_proba = model(x_test.to(device)).detach().cpu().numpy()  # 获取模型预测的概率
        y_pred = model(x_test.to(device)).argmax(dim=1).cpu().numpy()  # 获取模型预测的类别

        precision = precision_score(y_test.cpu().numpy().ravel(), y_pred)
        recall = recall_score(y_test.cpu().numpy().ravel(), y_pred)
        f1 = f1_score(y_test.cpu().numpy().ravel(), y_pred)
        auc = roc_auc_score(y_test.cpu().numpy().ravel(), y_pred_proba[:, 1])

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'AUC: {auc:.4f}')

        # 绘制ROC曲线和PR曲线
        fpr, tpr, _ = roc_curve(y_test.cpu().numpy().ravel(), y_pred_proba[:, 1])
        precision, recall, _ = precision_recall_curve(y_test.cpu().numpy().ravel(), y_pred_proba[:, 1])

        plt.figure(figsize=(10, 5))

        # 绘制ROC曲线
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='blue', label='ROC Curve')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

        # 绘制PR曲线
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='green', label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    main()
