import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from cartpole_with_bins import plot_running_avg

class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 10e-2

    def partial_fit(self, X, Y):
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)

class FeatureTransformer:
    def __init__(self, env):
        # Instead of using the observations examples given by the
        # environment (since some intervals are infinite),
        # We use values that we think can cover the most examples
        observation_examples = np.random.random((20000, 4)) * 2 - 1
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=0.05, n_components=1000)),
            ('rbf2', RBFSampler(gamma=0.1, n_components=1000)),
            ('rbf3', RBFSampler(gamma=0.5, n_components=1000)),
            ('rbf4', RBFSampler(gamma=1, n_components=1000))
        ])

        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

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
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self, s, a, G):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(model, eps, gamma):
    env = model.env
    observation = env.reset()
    done = False
    total_reward = 0
    iters = 0

    while not done and iters < 2000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        total_reward += reward

        if done:
            reward = -200

        # update the model (Q Learning equation)
        G = reward + gamma * np.max(model.predict(observation))
        model.update(prev_observation, action, G)

        iters += 1

    return total_reward


# env = gym.make('CartPole-v0')
# ft = FeatureTransformer(env)
# model = Model(env, ft)
# gamma = 0.99
#
#
# N = 500
# total_rewards = np.empty(N)
# for n in range(N):
#     eps = 1.0 / np.sqrt(n + 1)
#     total_reward = play_one(model, eps, gamma)
#     total_rewards[n] = total_reward
#     if n % 100 == 0:
#         print('episode:', n, 'total reward:', total_reward, 'eps:', eps)
# print('avg reward for last 100 episodes:', total_rewards[-100:].mean())
# print('total steps:', total_rewards.sum())
#
# plt.plot(total_rewards)
# plt.title('Rewards')
# plt.show()
#
# plot_running_avg(total_rewards)
#
# env = wrappers.Monitor(env, 'cartpole_videos')
# play_one(model, 0, 0.99)

