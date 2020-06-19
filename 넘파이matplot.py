import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
x = np.array([1.0, 2.0, 3.0])

print(type(x))

y = np.array([2.0, 4.0, 6.0])

print(x+y)
print(x-y)
print(x*y)
print(x/y)

print(x/2.0)

A = np.array([[1, 2], [3, 4]])
print(A)

print(A.shape, A.dtype)

B = np.array([10, 20])
A * B
print(A*B)

x = np.arange(0, 6, 0.1)
y = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")  # plot이 그래프를 그리는 부분
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()  # 라벨을 입력한 표지판(?)을 보여주는 부분
plt.show()  # 그래프를 보여주는 부분

# 이미지파일 보여주기
# img = imread('img.png')
# plt.imshow(img)
# plt.show()
