import im2col
import sys
import os
sys.path.append(os.pardir)

# im2col(input_data, filter_h, filter_w, stride, pad)

x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)  # (9,75)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)

print(col2.shape)  # (90,75)

# 이때 출력되는 75의 크기는 필터의 원소수와 같다(3*5*5)


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H+2*self.pad - FH) / self.stride)
        out_w = int(1 + (W+2*self.pad-FW)/self.stride)

    # ##########################
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1)).T  # 필터 전개, reshape -1을 지정하면
                        # 다차원 배열에서 원소수가 변환 후에도 똑같이 유지되도록 묶어준다.
                        # 즉 (10,3,5,5)는 총 750개의 원소인데 reshape(10,-1)을 하면 (10,75)로 묵어주는 것이다.
        out=np.dot(col, col_w) + self.b
    # 중요한 부분

        out=out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # transpose함수를 이용하는데 이는 axis의 순서를 바꿔주는 것이다.
        # (N,H,W,C)/ 0,1,2,3 ==transpose ==> (N,C,H,W) / 0,3,1,2
        return out


class Pooling:
    def __init__(self, poll_h, pool_w, stride = 1, pad = 0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad

    def forward(self, x):
        N, C, H, W=x.shape
        out_h=int(1+(H-self.pool_h)/self.stride)
        out_w=int(1 + (W-self.pool_w)/self.stride)

        # 전개1
        col=im2col(x, self.pool_h, self.stride, self.pad)
        col=col.reshape(-1, self.pool_h*self.pool_w)

        # 최댓값2
        out=np.max(col, axis = 1)

        # 성형
        out=out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out
