import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import gym
from cartpole_q_learning import plot_running_avg, play_one

class NN:
    def __init__(self, env, D):
        lr = 0.1

        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        # scaler = MinMaxScaler()
        # scaler.fit(observation_examples)
        # self.scaler = scaler

        # Initializa variable and placeholders
        self.w_in_hid = tf.Variable(tf.random_normal(shape=(D, 2000)) * tf.sqrt(2 / (2000 + D)), name='w_in_hid')
        self.b_hid = tf.Variable(tf.random_normal(shape=(1, 2000))* tf.sqrt(2 / (2000 + D)), name='b_hid')

        self.w_hid_out = tf.Variable(tf.random_normal(shape=(2000, 1)) * tf.sqrt(2 / 2001), name='w_hid_out')
        self.b_out = tf.Variable(tf.random_normal(shape=(1, 1)) * tf.sqrt(2 / (2001)), name='b_out')

        self.X = tf.placeholder(dtype=tf.float32, shape=(1, D), name='X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=(None,), name='Y')

        # Define operations
        Z_1 = tf.matmul(self.X, self.w_in_hid) + self.b_hid
        A_1 = tf.nn.relu(Z_1)
        Z_out = tf.matmul(A_1, self.w_hid_out) + self.b_out
        Y_out = tf.reshape(tf.nn.relu(Z_out), [-1])
        delta = self.Y - Y_out
        cost = tf.reduce_sum(delta * delta)

        # Training operations
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = Y_out

        # start session and initialize params
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit(self, X, Y):
        # X = self.scaler.transform(X)
        self.session.run(self.train_op, feed_dict={self.X:X, self.Y:Y})

    def predict(self, X):
        # X = self.scaler.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X:X})

class Model:
    def __init__(self, env):
        self.env = env
        self.models = []
        for i in range(env.action_space.n):
            model = NN(env, env.observation_space.shape[0])
            self.models.append(model)

    def predict(self, s):
        return np.array([m.predict([s]) for m in self.models])

    def update(self, s, a, G):
        self.models[a].partial_fit([s], [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


print('CARTPOLE WITH TENSORFLOW AND NEURAL NETWORKS')
print('---------------------------------------------\n')
env = gym.make('CartPole-v0')
model = Model(env)
gamma = 0.99


N = 50000
total_rewards = np.empty(N)
for n in range(N):
    eps = 1.0 / np.sqrt(n + 1)
    total_reward = play_one(model, eps, gamma)
    total_rewards[n] = total_reward
    if n % 1000 == 0:
        print('episode:', n, 'total reward:', total_reward, 'eps:', eps)
print('avg reward for last 100 episodes:', total_rewards[-100:].mean())
print('total steps:', total_rewards.sum())

plt.plot(total_rewards)
plt.title('Rewards')
plt.show()

plot_running_avg(total_rewards)