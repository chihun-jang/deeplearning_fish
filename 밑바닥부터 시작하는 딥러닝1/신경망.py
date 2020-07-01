
# 계단함수 구현하기


import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

# 위의 함수는 실수는 받아들일 수 있지만 배열을 arg로 받을 수는 없다.
# 따라서 배열을 받을수있게 수정해준다면


def step_function2(x):
    y = x > 0
    return y.astype(np.int)

# 위와같이 처리해주면 배열의 각각의 원소를 부등호로 판별해 배열을 리턴하는데
# 그 리턴된 배열의 True, false값을 int로 변환해서 return하는 모양이다.


def step_function3(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1+np.exp(-x))


# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function3(x)
# y2 = sigmoid(x)
# plt.plot(x, y)
# plt.plot(x, y2,  linestyle="--")

# plt.ylim(-0.1, 1.1)
# plt.show()

# ReLU함수 구현하기


def relu(x):
    return np.maximum(0, x)


A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))  # ndim으로 배열의 차원수 확인가능하고
print(A.shape)  # shape로 모양을 알수있고,(그런데 1차원 배열도 튜플로 반환하는 이유는 다차원 배열과 일치시켜주기 위해)
print(A.shape[0])


print("*"*50)
X = np.array([1, 2])
print(X.shape)
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W.shape)
Y = np.dot(X, W)
print(Y)

# 이처럼 행렬을 계산하는데 원소의 갯수와 상관없이 dot연산자를 쓰면 된다.

print("*"*50)

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])


A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(A1)
print("시그모이드 처리:", Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 마지막 출력층 구현하기 (다 비슷하지만 활성화 함수만 다름)
# 이때 출력층의 활성화함수는 문제에 맞춰 다르게 설정하게 되고, (회귀 문제는 항등함수, 이진 분류는 시그모이드, 다중분류는 softmax분류를 사용한다.)


def identity_function(x):
    return x


W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)


print(Y)


# softmax함수 구현하기
print("#"*100)
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a/sum_exp_a
print(y)


def softmax(a):
    exp_a = np_exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
# softmax함수를 구현할때오버플로가 일어나는 부분을 조심해야하는데 추가 처리를 해줘야한다


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)  # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


a = np.array([0.3, 2.9, 4.0])

y = softmax(a)
print(y)
print(np.sum(y))
