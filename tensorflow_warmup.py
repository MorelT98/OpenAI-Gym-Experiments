import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
import matplotlib.pyplot as plt
from cartpole_q_learning import FeatureTransformer, play_one, plot_running_avg

class SGDRegressor:
    def __init__(self, D):
        lr = 0.1
        # matmul doesn't like when the matrix is 1D, so we use a 2D matrix for w and X
        self.w = tf.Variable(tf.random_normal(shape=(D, 1)), name='w')
        self.X = tf.placeholder(dtype=tf.float32, shape=(1, D), name='X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=(None,), name='Y')

        # Prediction and cost
        Y_hat = tf.reshape(tf.matmul(self.X, self.w), [-1])
        delta = self.Y - Y_hat
        cost = tf.reduce_sum(delta * delta)

        # operations useful for later
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = Y_hat

        # start session and initialize params
        init = tf.global_variables_initializer()
        # we use interactiveSession to be able to use the same session
        # for different functions
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X:X, self.Y:Y})

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X:X})

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer
        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(feature_transformer.dimensions)
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        return np.array([m.predict(X) for m in self.models])

    def update(self, s, a, G):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


print('CARTPOLE WITH TENSORFLOW')
print('-------------------------\n')
env = gym.make('CartPole-v0')
ft = FeatureTransformer(env)
model = Model(env, ft)
gamma = 0.99


N = 500
total_rewards = np.empty(N)
for n in range(N):
    eps = 1.0 / np.sqrt(n + 1)
    total_reward = play_one(model, eps, gamma)
    total_rewards[n] = total_reward
    if n % 100 == 0:
        print('episode:', n, 'total reward:', total_reward, 'eps:', eps)
print('avg reward for last 100 episodes:', total_rewards[-100:].mean())
print('total steps:', total_rewards.sum())

plt.plot(total_rewards)
plt.title('Rewards')
plt.show()

plot_running_avg(total_rewards)

