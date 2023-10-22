import numpy as np
from inverse_dynamics import RealInvNetwork
from transition_function import reshapeaction_transition,reshapeaction,inversereshapeaction
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.autograd import grad
import scipy
import matplotlib.pyplot as plt
# from ar_train import policy_evaluation

k1=0.5
k2=0.5
p=0.2

net = RealInvNetwork()
net.load_state_dict(torch.load('RealInvNetwork.pth'))

# 创建一个从0到1之间的一系列x值作为e
batch_size = 1000
e = np.linspace(0, 1, batch_size)

# 对应的a = f(e)
a = [reshapeaction(xi,k1,k2,p)[0] for xi in e]
J_a = [reshapeaction(xi,k1,k2,p)[1] for xi in e]

# 输入e,在当前参数alpha,beta下的s'
transition_value = [reshapeaction_transition(xi,k1,k2,p) for xi in e]

# 输入e,在当前参数alpha=1.0,beta=1.0下的结果s'
transition_value_2 = [reshapeaction_transition(xi) for xi in e]


input_data = np.zeros((batch_size, 2), dtype=np.float32)
input_data[:, 0] = 0
input_data[:, 1] = transition_value

input_data = torch.from_numpy(input_data)

mu = net(input_data)[0].squeeze()
sigma = net(input_data)[1].squeeze()

a_tensor = torch.tensor(a)
normal_log_likelihood = -0.5 * ((a_tensor - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(np.pi))
# normal_log_likelihood = torch.clip(normal_log_likelihood,min = -10)
# \log |\frac{\partial f(\theta, e)}{\partial e}|
#  & =  \log \left(\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha) \Gamma(\beta)} e^{\alpha-1}(1-e)^{\beta-1}\right) \\
# & =\log \Gamma(\alpha+\beta)-\log \Gamma(\alpha)-\log \Gamma(\beta)+(\alpha-1) \log e +(\beta-1) \log (1 - e)
epsilon = 1e-8
# beta_log = scipy.special.gammaln(alpha + beta) - scipy.special.gammaln(alpha) - scipy.special.gammaln(beta) \
#            + (alpha - 1)*np.log(e + epsilon) + (beta -1) * np.log(1-e+epsilon)
beta_log = np.array(J_a)
J = - normal_log_likelihood.mean().item() - beta_log.mean()

J2 = 0.0
# _,_,J2 = policy_evaluation(alpha,beta,batch_size,net)

# 对a的预测
predict_mean = mu.tolist()
predict_var = sigma.tolist()

# 对e的预测
predict_e_mean = [inversereshapeaction(xi,k1,k2,p)[0] for xi in predict_mean]


# 创建一个图形窗口
plt.figure(figsize=(20, 5))


plt.subplot(1, 4, 1)
plt.plot(a, predict_mean, label='predict_mean')
plt.plot(a, a, label='real_mean')
plt.xlabel('a')
plt.ylabel('predict_mean')
plt.title('predict_mean k1:' + str(k1)+" k2:" +str(k2)+" p:"+str(p))
plt.grid(True)
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(e, predict_e_mean, label='predict_e_mean')
plt.plot(e, e, label='real_e_mean')
plt.xlabel('e')
plt.ylabel('predict_e_mean')
plt.title('predict_mean k1:' + str(k1)+" k2:" +str(k2)+" p:"+str(p))
plt.grid(True)
plt.legend()

plt.subplot(1, 4, 3)
plt.plot(e, predict_var, label='v')
plt.xlabel('a')
plt.ylabel('predict_var')
plt.title('predict_var')
plt.grid(True)
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(e, a, label = "f function")
plt.plot(e, transition_value_2, label='trans_a')
plt.plot(e, transition_value, label='trans_e')
plt.xlabel('a/e')
plt.ylabel('s_prime')
plt.title('trans')
plt.grid(True)
plt.legend()

# 显示图形
plt.title("J_target:  {:.2f} J_target_2:  {:.2f}".format(J,J2))
plt.tight_layout()
plt.show()

