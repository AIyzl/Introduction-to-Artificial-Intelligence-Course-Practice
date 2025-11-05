import pygame  # 导入相应的包
import sys
import torch
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import serial  # Python与单片机通信
from pylsl import StreamInfo, StreamOutlet   # 用于与LSL进行交互
rawData = serial.Serial('COM4'
                        '', 4800)  # 配置串口
info = StreamInfo('EDUduino', 'EEG', 1, 512, 'float32', 'myuid34234')   # 设置LSL数据流
outlet = StreamOutlet(info)    # 创建一个输出对象，该对象用于将数据推送到LSL数据中
def get_attention():
    raw = []
    processed = []
    out = []
    count = 0
    K = True
    while K:
        if (rawData.read() == b'\xaa'):
            if (rawData.read() == b'\xaa'):
                c = rawData.read()
                if c == b'\x04':
                    # 读取下一个字节并检查其是否为 0x80
                    if rawData.read() == b'\x80':
                        # 读取下一个字节并检查其是否为 0x02
                        if rawData.read() == b'\x02':
                            # 从串口读取两个字节，并将它们转换为一个整数值，用于构造一个16位的数据值
                            rawdata1 = ord(rawData.read()) * 256 + ord(rawData.read())
                            # 如果数据值超过了一个有符号的16位整数所能表示的范围（即大于32768），
                            # 则减去65536，以得到正确的有符号整数值。
                            if rawdata1 > 32768:
                                rawdata1 = rawdata1 - 65536
                                # print(rawdata1)
                            # 将处理后的数据值推送到 LSL（Lab Streaming Layer）数据流中
                            outlet.push_sample([rawdata1])
                            raw.append(rawdata1)
                            count = count + 1
                            # 如果重新循环了512*second次，则结束循环
                            if count == 512 * 3:
                                K = False
                if (c == b' '):  # ord(b'\x20')==ord(b' ')==32
                    if (rawData.read() == b'\x02'):
                        rawData.read()
                        if (rawData.read() == b'\x83'):
                            if (rawData.read() == b'\x18'):
                                # 将从串口接收到的三个字节的数据解析为一个整数值
                                # 计算脑电波（delta、theta、lowalpha、highalpha、lowbeta、highbeta、lowgamma、middlegamma）数值
                                delta = ord(rawData.read()) * 256 * 256 + ord(rawData.read()) * 256 + \
                                        ord(rawData.read())
                                # print("delta=")
                                # print(delta)
                                # 处理后的数据值添加到名为processed的列表中，下同
                                processed.append(delta)
                                theta = ord(rawData.read()) * 256 * 256 + ord(rawData.read()) * 256 + \
                                        ord(rawData.read())
                                # print("theta=")
                                # print(theta)
                                processed.append(theta)
                                lowalpha = ord(rawData.read()) * 256 * 256 + ord(rawData.read()) * 256 + \
                                           ord(rawData.read())
                                # print("lowalpha=")
                                # print(lowalpha)
                                processed.append(lowalpha)

                                highalpha = ord(rawData.read()) * 256 * 256 + ord(rawData.read()) * 256 + \
                                            ord(rawData.read())
                                # print("highalpha=")
                                # print(highalpha)
                                processed.append(highalpha)

                                lowbeta = ord(rawData.read()) * 256 * 256 + ord(rawData.read()) * 256 + \
                                          ord(rawData.read())
                                # print("lowbeta=")
                                # print(lowbeta)
                                processed.append(lowbeta)

                                highbeta = ord(rawData.read()) * 256 * 256 + ord(rawData.read()) * 256 + \
                                           ord(rawData.read())
                                # print("highbeta=")
                                # print(highbeta)
                                processed.append(highbeta)

                                lowgamma = ord(rawData.read()) * 256 * 256 + ord(rawData.read()) * 256 + \
                                           ord(rawData.read())
                                # print("lowgamma=")
                                # print(lowgamma)
                                processed.append(lowgamma)

                                middlegamma = ord(rawData.read()) * 256 * 256 + ord(rawData.read()) * 256 + \
                                              ord(rawData.read())
                                # print("middlegamma=")
                                # print(middlegamma)
                                processed.append(middlegamma)
                                out.append(processed)
                                processed = []
                                # if (rawData.read() == b'\x04'):
                                #     attention = ord(rawData.read())
                                #     print('注意力：{}'.format(attention))

    return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构建模型
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN_LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=input_size, kernel_size=3)  # 输出通道数改为input_size
        self.maxpool = nn.MaxPool1d(kernel_size=2)

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

def model_use(a):
    count = 0
    sc = StandardScaler()
    a = np.array(a)
    a = sc.fit_transform(a)
    a = torch.tensor(a, dtype=torch.float32)
    a = a.unsqueeze(1)
    model_path = './model_CNN_LSTM.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    the_model = torch.load(model_path, map_location=device)
    # 输出测试集的预测标签值
    the_model.eval()
    with torch.no_grad():
        a_out = the_model(a.to(device)).argmax(dim=1)
        # print(a_out.cpu().numpy())
    for i in a_out:
        if i == 1:
            count = count + 1
    return count
def result():
    a = get_attention()
    b = model_use(a)
    # print(b)
    return b

# 定义颜色
attention = 0
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
pygame.init()  # 初始化
# 注意力反馈
font = pygame.font.SysFont('simHei', 20)
text = font.render("注意力", True, BLACK)
text1 = font.render("集中", True, RED)
text2 = font.render("分散", True, BLUE)
pygame.display.set_caption('基于注意力机制的虚拟小车')
screen = pygame.display.set_mode((462, 684))  # 绘制界面
way = pygame.image.load('way.jpg')  # 导入界面背景
# 设置图像帧率
FPS = 120
clock = pygame.time.Clock()
# 定义一个控制虚拟小车的类
class Car:
    def __init__(self):
        centerx, centery = 236, 315
        # 导入虚拟小车
        self.image = pygame.image.load('img.png')
        # 小车位置初始化
        self.rect = self.image.get_rect(center=(centerx, centery))   # (0, 0)

    def move(self):
        # 控制虚拟小车
        global attention
        mark = result()
        pressed_keys = pygame.key.get_pressed()
        if mark > 1:
            # 设置前进的边界条件
            if car.rect.centery >= 70:
                screen.blit(text1, (70, 25))
                self.rect.move_ip(0, -25)

        if mark <= 1:
            # 设置后退的边界条件
            if car.rect.centery <= 632:
                screen.blit(text2, (70, 25))
                self.rect.move_ip(0, 25)

        if pressed_keys[pygame.K_LEFT]:
            # 设置左转的边界条件
            if car.rect.centerx >= 76:
                self.rect.move_ip(-2, 0)
                # attention += 1

        if pressed_keys[pygame.K_RIGHT]:
            # 设置右转的边界条件
            if car.rect.centerx <= 380:
                self.rect.move_ip(2, 0)
car = Car()
while True:
    # 加载虚拟小车和运动背景
    screen.blit(way, (0, 0))
    screen.blit(car.image, car.rect)
    screen.blit(text, (10, 25))
    car.move()
    # 捕捉当前事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # 结束条件
            pygame.quit()
            sys.exit()
    # 实时更新界面
    pygame.display.update()
    clock.tick(FPS)
