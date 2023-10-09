import numpy as np
import matplotlib.pyplot as plt
import scipy

def reshapeaction_transition(e,alpha=1.0,beta=1.0,randomness=0.0):
    a = scipy.special.betaincinv(alpha, beta, e)
    acceleration = (a <0.5) * 0 + (a >= 0.5) *((a - 0.5) / 0.5)
    acceleration += randomness * np.random.normal(size=a.size)
    return acceleration

# def reshapeaction_transition(e,alpha=1.0,beta=1.0,randomness=0.0):
#     a = scipy.special.betaincinv(alpha, beta, e)
#     acceleration = (a <0.5) * 2*a + (a >= 0.5) * 1
#     acceleration += randomness * np.random.normal(size=a.size)
#     return acceleration

if __name__ == '__main__':
    # 定义Beta分布的形状参数
    a = 0.17
    b = 0.99

    # 创建一个从0到1之间的一系列x值
    x = np.linspace(0, 1, 100)

    # 计算不完全Beta函数（betainc）的值
    betainc_values = [scipy.special.betainc(a, b, xi) for xi in x]

    # 计算不完全Beta函数的逆（betaincinv）的值
    betaincinv_values = [scipy.special.betaincinv(a, b, xi) for xi in x]

    # 创建一个图形窗口
    plt.figure(figsize=(15, 5))

    # 绘制不完全Beta函数（betainc）的图像
    plt.subplot(1, 3, 1)
    plt.plot(x, betainc_values, label='e = h(a)')
    plt.xlabel('x')
    plt.ylabel('h function')
    plt.title('h function')
    plt.grid(True)
    plt.legend()

    # 绘制不完全Beta函数的逆（betaincinv）的图像
    plt.subplot(1, 3, 2)
    plt.plot(x, betaincinv_values, label='a = f(e)')
    plt.xlabel('x')
    plt.ylabel('f function')
    plt.title('f function')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    transition_value = [reshapeaction_transition(xi) for xi in x]
    plt.plot(x, transition_value, label='trans')
    plt.xlabel('a')
    plt.ylabel('s_prime')
    plt.title('trans')
    plt.grid(True)
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()

