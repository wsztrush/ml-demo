import tensorflow as tf
# 变量在创建时并不会赋值
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# 定义线性模型
linear_model = W * x + b
# 定义误差，也就是方差
loss = tf.reduce_sum(tf.square(linear_model - y))
# 定义优化方式，通过梯度下降来减小误差
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
with tf.Session() as sess:
    # 为变量赋值（直到这里变量才能用。。。）
    sess.run(tf.global_variables_initializer())
    # 迭代计算
    for i in range(1000):
        sess.run(train, {x: [0, 1, 2, 3], y: [20.76150322, 4, 25.06471634, 12.75303841]})
    # 打印参数
    print(sess.run([W, b]))  # [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]