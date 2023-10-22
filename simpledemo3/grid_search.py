import numpy as np
from inverse_dynamics import RealInvNetwork
from transition_function import reshapeaction_transition
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.autograd import grad
import scipy
import matplotlib.pyplot as plt
from ar_train import policy_evaluation

alpha_min = 1.2
beta_min = 1.2
alpha_max = 5.2
beta_max = 5.2

parameter_interval = 0.2
alpha_num = int(np.round((alpha_max-alpha_min)/parameter_interval + 1))
beta_num = int(np.round((beta_max-beta_min)/parameter_interval + 1))
J_matrix = np.zeros((alpha_num,beta_num))
J_alpha_matrix = np.zeros((alpha_num,beta_num))
J_beta_matrix = np.zeros((alpha_num,beta_num))

net = RealInvNetwork()
net.load_state_dict(torch.load('RealInvNetwork.pth'))
batch_size = 100

for i in range(alpha_num):
    for j in range(beta_num):
        alpha = alpha_min + i * parameter_interval
        beta = beta_min + j * parameter_interval

        gradient_alpha,gradient_beta,J2 = policy_evaluation(alpha,beta,batch_size,net)
        J_matrix[i,j] = J2
        J_alpha_matrix[i,j] = gradient_alpha
        J_beta_matrix[i,j] = gradient_beta
        print(f"alpha: {alpha:.2f} beta: {beta:.2f} Target: {J2:.2f}")




# 绘制热力图
# fig, ax = plt.subplots()
plt.figure(1)
matrix = J_matrix
plt.imshow(matrix.transpose(),cmap='binary')

# 添加坐标轴
plt.xticks(np.arange(alpha_num),np.around(np.arange(alpha_min,alpha_max+parameter_interval,parameter_interval),2))
plt.yticks(np.arange(beta_num),np.around(np.arange(beta_min,beta_max+parameter_interval,parameter_interval),2))
plt.ylabel("alpha")
plt.xlabel("beta")

# 添加注释
for i in range(len(matrix[0])):
    for j in range(len(matrix)):
        text = plt.text(j, i, f"{matrix[i, j]:.0f}", ha='center', va='center', color='red')

plt.colorbar()  # 添加颜色条
plt.show()

plt.figure(2)
matrix = J_alpha_matrix
plt.imshow(matrix.transpose(),cmap='binary')

# 添加坐标轴
plt.xticks(np.arange(alpha_num),np.around(np.arange(alpha_min,alpha_max+parameter_interval,parameter_interval),2))
plt.yticks(np.arange(beta_num),np.around(np.arange(beta_min,beta_max+parameter_interval,parameter_interval),2))
plt.ylabel("alpha")
plt.xlabel("beta")

# 添加注释
for i in range(len(matrix[0])):
    for j in range(len(matrix)):
        text = plt.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color='red')

plt.colorbar()  # 添加颜色条
plt.show()

plt.figure(3)
matrix = J_beta_matrix
plt.imshow(matrix.transpose(),cmap='binary')

# 添加坐标轴
plt.xticks(np.arange(alpha_num),np.around(np.arange(alpha_min,alpha_max+parameter_interval,parameter_interval),2))
plt.yticks(np.arange(beta_num),np.around(np.arange(beta_min,beta_max+parameter_interval,parameter_interval),2))
plt.ylabel("alpha")
plt.xlabel("beta")

# 添加注释
for i in range(len(matrix[0])):
    for j in range(len(matrix)):
        text = plt.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color='red')

plt.colorbar()  # 添加颜色条
plt.show()




