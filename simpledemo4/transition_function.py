import numpy as np
import matplotlib.pyplot as plt
import scipy

def reshapeaction_transition(e,alpha=1.0,beta=1.0,randomness=0.01):
    a = scipy.special.betainc(alpha, beta, e)
    k = 2
    acceleration = (a < (0.5/k)) * k * a + (a > (1 - 0.5/k)) *(k*a +1-k) + (a>=(0.5/k)) * (a <= (1 - 0.5/k)) * 0.5
    acceleration += randomness * np.random.normal(size=a.size)
    return acceleration

# def reshapeaction_transition(e,alpha=1.0,beta=1.0,randomness=0.01):
#     a = scipy.special.betainc(alpha, beta, e)
#     acceleration = (a <0.5) * 0 + (a >= 0.5) *((a - 0.5) / 0.5)
#     acceleration += randomness * np.random.normal(size=a.size)
#     return acceleration

# def reshapeaction_transition(e,alpha=1.0,beta=1.0,randomness=0.01):
#     a = scipy.special.betainc(alpha, beta, e)
#     acceleration = (a <0.5) * 2*a + (a >= 0.5) * 1
#     acceleration += randomness * np.random.normal(size=a.size)
#     return acceleration

if __name__ == '__main__':
    # 创建一个从0到1之间的一系列x值
    x = np.linspace(0, 1, 100)

    # 创建一个图形窗口
    plt.figure(figsize=(5, 5))

    plt.subplot(1, 1, 1)
    transition_value = [reshapeaction_transition(xi) for xi in x]
    plt.plot(x, transition_value, label='original_trans_a')
    plt.xlabel('a')
    plt.ylabel('s_prime')
    plt.title('trans')
    plt.grid(True)
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()

