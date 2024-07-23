import paddle


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
    tmp = paddle.inverse(paddle.matmul(x_sub.T,x_sub) + reg_lambda * paddle.eye(num_rows =D))
    w = paddle.matmul(paddle.matmul(tmp,x_sub.T), (y-y_bar))
    b = y_bar-paddle.matmul(x_bar_tran,w)
    model.params['b'] = b
    model.params['w'] = paddle.squeeze(w,axis=-1)
    return model
