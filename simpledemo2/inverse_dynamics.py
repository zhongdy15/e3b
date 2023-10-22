import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transition_function import reshapeaction_transition
import scipy

# 修改Realinvnetwork为预测a，这与alpha beta无关
class RealInvNetwork(nn.Module):
    def __init__(self):
        super(RealInvNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 128)  # 第一个额外的全连接层
        self.fc3 = nn.Linear(128, 64)  # 第二个额外的全连接层
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # 使用ReLU激活函数添加额外的层
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        sigma = torch.ones_like(mu)*0.015
        return mu, sigma

if __name__ == '__main__':
    # 创建神经网络实例
    net = RealInvNetwork()

    # 定义批处理大小和训练周期数
    batch_size = 256
    num_epochs = 1000

    # 创建一个优化器（例如，Adam优化器）
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 开始训练循环
    for epoch in range(num_epochs):
        total_loss = 0.0


        s = 0 # 初始加速度是0

        # 均匀分布采集a
        a = np.random.uniform(0, 1, batch_size)

        s_prime = reshapeaction_transition(a,alpha=1.0,beta=1.0)

        # 将输入数据包装成 PyTorch 张量
        input_data = np.zeros((batch_size, 2), dtype=np.float32)
        input_data[:, 0] = s
        input_data[:, 1] = s_prime

        input_data = torch.from_numpy(input_data)

        target_data = np.zeros((batch_size, 1), dtype=np.float32)
        target_data[:, 0] = a
        target_data = torch.from_numpy(target_data)

        optimizer.zero_grad()
        mu, sigma = net(input_data)

        # 使用reparameterization trick来采样
        epsilon = torch.randn_like(mu)
        sampled_value = mu + epsilon * sigma

        loss = nn.functional.mse_loss(sampled_value, target_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # if epoch > 130:
        #     print("test")

        avg_loss = total_loss
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

    # 训练完成后，您可以保存模型
    torch.save(net.state_dict(), 'RealInvNetwork.pth')
