import paddle


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


y_true = paddle.to_tensor([[-0.2], [4.9]], dtype='float32')
y_pred = paddle.to_tensor([[1.3], [2.5]], dtype="float32")

error = mean_squared_error(y_true=y_true, y_pred=y_pred).item()  # Tensor的作用是得到向量对应的值
print("error:",error)