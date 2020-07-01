class MulLayer:
    # 객체변수인 x와 y를 초기화한다.
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backward(self, dout):
        dx = dout*self.y
        dy = dout * self.x
        return dx, dy

# 사과 예제를 MulLayer를 사용하여 풀어보기


# apple = 100
# apple_num = 2
# tax = 1.1

# mul_apple_layer = MulLayer()
# mul_tax_layer = MulLayer()
# apple_price = mul_apple_layer.forward(apple, apple_num)
# price = mul_tax_layer.forward(apple_price, tax)

# print(price)

# # 각 변수에 대한 미분은 backward에서 구하자

# dprice = 1
# dapple_price, dtax = mul_tax_layer.backward(dprice)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# print(dapple, dapple_num, dtax)
# backward()가 받는 인수는 순전파의 출력에 대한 미분


class AddLayer:
    # 덧셈계층은 x, y가 굳이 필요없으니 init이 아무일도 하지않는다
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x+y
        return out

    # backward() 에서는 출력으로부터 돌아오는 값을 그래도 다시 흘려보낸다
    # 상류 -> 하류로
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)

all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price)
print(dapple_num, dapple, dorange, dorange_num, dtax)


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # x는 넘파이 배열로 들어오는데 x가 0보다 작거나 같은 index를 True로 하여 배열생성
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0  # 0보다 작거나 같은 애들이 True이므로 그 index에 해당하는 ele를 0으로 바꿈

        return out

    # 역전파는 순전파때 만든 mask를 이용해서 True인 index에 대해서 dout을 0으로 전달한다.
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

# ReLU는 전형적인 전기회로 문제와 비슷하다 전류가 흐를때는 전류를 흘려보내고
# 스위치가 꺼지면 전류가 아예 안흘러가는


class Sigmoid:
    def __init__(self):
        self.out = None

    # 순전파의 출력을 out에 보관하다가 역전파 계산때 값을 사용한다.
    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out)*self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실
        self.y = None  # softmax의 출력
        self.t = None  # 정답레이블(원핫인코딩)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        # 역전파로 전파하는 값을 배치의 수로 나눠 data1개당 오차를 앞으로 전파
        dx = (self.y - self.t) / batch_size

        return dx
