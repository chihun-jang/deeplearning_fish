from gradient import numerical_gradient as real_num_grad
from functions import softmax, cross_entropy_error as real_cross_entropy
import matplotlib.pylab as plt
from mnist import load_mnist
import numpy as np
import sys
import os
sys.path.append(os.pardir)
# 오차제곱합 cost function


def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
result = sum_squares_error(np.array(y), np.array(t))
print(result)
result2 = sum_squares_error(np.array(y2), np.array(t))
print(result2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))


a = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
b = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
b2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
result3 = cross_entropy_error(np.array(b), np.array(a))
print(result3)
result4 = cross_entropy_error(np.array(b2), np.array(a))
print(result4)
# log를 계산할때 작은 delta값을 더해줬다. 이는 np.log() 함수에 0이 들어가면 계산을 진행할수없기에 아주 작은 값이라도 더해
# 0이 안되게 해주는것이다.


(x_train, t_train), (x_test, t_test) =\
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


def cross_entropy_error2(y, t):
    if y.ndim == 1:  # 데이터가 1개일경우
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size
# y가 신경망 출력, t가 정답레이블 y가 1차원이면, 즉 데이터 1개당 크로스 엔트로필를 구하는 경우
# reshape로 data형상을 바꿔준다


def cross_entropy_error3_notonehot(y, t):
    if y.ndim == 1:  # 데이터가 1개일경우
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]+1e-7))/batch_size
# 여기서 핵심은 t가 0인 원소는 크로스 엔트로피도 0이므로 계산은 무시해도 된다.
# 다시말해서 정답에 해당하는 출력이랑만 비교하자

# np.log(y[np.arange(batch_size),t])란 무엇인가요
# np.arange는 0부터 batch_size전까지 배열을 생성한다.
# y[np.arange,t] 는 각 데이터의 정답 레이블에 해당하는 신경망 출력 추출
# 그러면 y[0,2], y[1,7] 이런식으로 배열이 생성됨


# 파이썬으로 미분 공식을 나타낸것, h는 작은 값을 임의로 대입해준다.
    # 이름은 수치미분으로 해준것.
def numerical_diff(f, x):
    h = 10*e - 50
    return (f(x+h)-f(x))/h

# 그런데 위에서 넣어준 임의의 h 값 10e-50 은 가수가 10이므로 소숫점 49자리 숫자인데
# 이 숫자는 rounding error문제를 일으킨다(소수점 몇자리 이하 반올림으로 계산결과에 오차가 생기는것,)


print(np.float32(1e-50))

# 따라서 적당히 10^-4정도만 해주도록 하자

# h의 값과 차분을 무한히 줄일수 없어서 나타나는 에러를 수정하여 다시 작성해보면


def numerical_diff2(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h))/(2*h)


# 이를 바탕으로 실습을 한번해보자

def function_1(x):
    return 0.01*x**2 + 0.1*x


# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# plt.show()


print(numerical_diff2(function_1, 5))

print(numerical_diff2(function_1, 10))
# 위 두값의 실제 값은 각각 0.2와 0.3인데 수치미분과 비교하면 오차가 매우 작음을 알수있따.


# 편미분
def function_2(x):
    return x[0]**2 + x[1] ** 2
    # 혹은 return np.sum(x**2)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # x와 같은 크기만큼 0로 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x+h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))

# 이렇게 해주면 각 점에서의 기울기를 계산해줄수 잇다.
# 그리고 위에서 구한 기울기, 벡터들로는 벡터장을 그려줄수있다.
# 그림 그려진 벡터장으로 확인읋 해보면 기울기(화살표)는 각 지점에서 낮아지는 방향을 가리키는데
# 이는 해당 방향으로 갈때 함수의 출력값이 줄어드는 것을 의미한다.


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
print("?", gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))

# 처음 x값을 3,4로 설정했는데 0.0에 정말 근사한 값으로 다가가는 모습을 확인할수 있따


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화

    # 예측하는 함수
    def predict(self, x):
        return np.dot(x, self.W)

    # 손실함수
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = real_cross_entropy(y, t)
        return loss


net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))

t = np.array([0, 0, 1])
print(net.loss(x, t))

# 기울기 구하기


def f(W):  # 직접적으로 W를 받고있지는 않지만 같은 instance의 W를 사용해서 내부적으로 메서드작업을 해주고있따.
    return net.loss(x, t)


dW = real_num_grad(f, net.W)
print(dW)

# 이러한 기울기를 보면 각 W에서의 중요도 및 양으로 움직일때 손실함수가 증가하는지 감소하는지 확인 할 수 있다.

# def f(W):  # 직접적으로 W를 받고있지는 않지만 같은 instance의 W를 사용해서 내부적으로 메서드작업을 해주고있다..
#     return net.loss(x, t)
# #위의 함수는 아래와 같이 람다함수로도 구현한다.
# f = lambda w: net.loss(x,t)
