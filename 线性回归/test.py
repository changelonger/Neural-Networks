# 数据集构建
def linear_func(x, w=1.2, b=0.5):
    y = w * x + b
    return y


import paddle


def create_toy_data(func, interval, sample_num, noise=0.0, add_outlier=False, outlier_ratio=0.001):
    """
    根据给定的函数，生成样本
    输入：
       - func：函数
       - interval： x的取值范围
       - sample_num： 样本数目
       - noise： 噪声均方差
       - add_outlier：是否生成异常值
       - outlier_ratio：异常值占比
    输出：
       - X: 特征数据，shape=[n_samples,1]
       - y: 标签数据，shape=[n_samples,1]
    """

    # 均匀采样
    # 使用paddle.rand在生成sample_num个随机数
    X = paddle.rand(shape=[sample_num]) * (interval[1] - interval[0]) + interval[0]
    y = func(X)

    # 生成高斯分布的标签噪声
    # 使用paddle.normal生成0均值，noise标准差的数据
    epsilon = paddle.normal(0, noise, paddle.to_tensor(y.shape[0]))
    y = y + epsilon
    if add_outlier:  # 生成额外的异常点
        outlier_num = int(len(y) * outlier_ratio)
        if outlier_num != 0:
            # 使用paddle.randint生成服从均匀分布的、范围在[0, len(y))的随机Tensor
            outlier_idx = paddle.randint(len(y), shape=[outlier_num])
            y[outlier_idx] = y[outlier_idx] * 5
    return X, y


func = linear_func
interval = (-10, 10)
train_num = 10000  # 训练样本数目
test_num = 5000  # 测试样本数目
noise = 2
X_train, y_train = create_toy_data(func=func, interval=interval, sample_num=train_num, noise=noise, add_outlier=False)
X_test, y_test = create_toy_data(func=func, interval=interval, sample_num=test_num, noise=noise, add_outlier=False)

X_train_large, y_train_large = create_toy_data(func=func, interval=interval, sample_num=5000, noise=noise,
                                               add_outlier=False)

# paddle.linspace返回一个Tensor，Tensor的值为在区间start和stop上均匀间隔的num个值，输出Tensor的长度为num
X_underlying = paddle.linspace(interval[0], interval[1], train_num)
y_underlying = linear_func(X_underlying)
X_underlying = X_underlying.numpy()
y_underlying = y_underlying.numpy()


# 损失函数
def mean_squared_error(y_true, y_pred):
    """
        输入：
           - y_true: tensor，样本真实标签
           - y_pred: tensor, 样本预测标签
        输出：
           - error: float，误差值
    """
    assert y_true.shape[0] == y_pred.shape[0]
    # paddle.square计算输入的平方值
    # paddle.mean沿 axis 计算 x 的平均值，默认axis是None，则对输入的全部元素计算平均值。
    error = paddle.mean(paddle.square(y_true - y_pred))
    return error


# 模型构建
import paddle
from nndl.op import Op

paddle.seed(10)  # 设置随机种子


# 线性算子
class Linear(Op):
    def __init__(self, input_size):
        """
        输入：
           - input_size:模型要处理的数据特征向量长度
        """

        self.input_size = input_size

        # 模型参数
        self.params = {}
        self.params['w'] = paddle.randn(shape=[self.input_size, 1], dtype='float32')
        self.params['b'] = paddle.zeros(shape=[1], dtype='float32')

    def __call__(self, X):
        return self.forward(X)

    # 前向函数
    def forward(self, X):
        """
        输入：
           - X: tensor, shape=[N,D]
           注意这里的X矩阵是由N个x向量的转置拼接成的，与原教材行向量表示方式不一致
        输出：
           - y_pred： tensor, shape=[N]
        """

        N, D = X.shape

        if self.input_size == 0:
            return paddle.full(shape=[N, 1], fill_value=self.params['b'])

        assert D == self.input_size  # 输入数据维度合法性验证

        # 使用paddle.matmul计算两个tensor的乘积
        y_pred = paddle.matmul(X, self.params['w']) + self.params['b']
        return y_pred


# 模型优化
def optimizer_lsm(model, X, y, reg_lambda=0):
    """
    输入：
        -model：模型
        -X:tensor,特征数据，shape=[N,D]
        -y:tensor,标签数据，shape=[N]
        -reg_lambda:float,正则化系数，默认是零
    输出：
        model:优化好的模型
    """
    N, D = X.shape
    # 输入特征向量的平均值
    x_bar_tran = paddle.mean(X, axis=0)

    # 标签值均值
    y_bar = paddle.mean(y)

    # paddle.subtract通过广播的方式实现矩阵的减向量
    x_sub = paddle.subtract(X, x_bar_tran)

    # 使用paddle.all判断输入的tensor是否是零
    if paddle.all(x_sub == 0):
        model.params['b'] = y_bar
        model.params['w'] = paddle.zeros(shape=[D])
        return model
    # paddle.inverse求方阵的逆
    tmp = paddle.inverse(paddle.matmul(x_sub.T, x_sub) + reg_lambda * paddle.eye(num_rows=D))
    w = paddle.matmul(paddle.matmul(tmp, x_sub.T), (y - y_bar))
    b = y_bar - paddle.matmul(x_bar_tran, w)
    model.params['b'] = b
    model.params['w'] = paddle.squeeze(w, axis=-1)
    return model


input_size = 1
model = Linear(input_size)
model = optimizer_lsm(model, X_train.reshape([-1, 1]), y_train.reshape([-1, 1]))
print("w_pred:", model.params['w'].item(), "b_pred: ", model.params['b'].item())

y_train_pred = model(X_train.reshape([-1, 1])).squeeze()
train_error = mean_squared_error(y_true=y_train, y_pred=y_train_pred).item()
print("train error: ", train_error)


y_test_pred = model(X_test.reshape([-1,1])).squeeze()
test_error = mean_squared_error(y_true=y_test, y_pred=y_test_pred).item()
print("test error: ",test_error)