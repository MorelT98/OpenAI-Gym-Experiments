import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gym import wrappers
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

class FeatureTransformer:
    def __init__(self, env):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()

        # Finds the means and stdevs for each feature (I assume)
        scaler.fit(observation_examples)

        # Gamma varies inversely with the variance, and n_components is the number
        # of kernels of the RBFSampler
        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=5, n_components=500)),
            ('rbf2', RBFSampler(gamma=2, n_components=500)),
            ('rbf3', RBFSampler(gamma=1, n_components=500)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=500))
        ])

        # Choose 500 kernels out of the 10000 examples, for each RBFSampler
        featurizer.fit(scaler.transform(observation_examples))

        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        # feature scaling
        scaled = self.scaler.transform(observations)
        # RBF transformation
        return self.featurizer.transform(scaled)

class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.feature_transformer = feature_transformer

        self.models = []
        # Create a linear model for each action
        for i in range(self.env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            # Initializa the model, to be able to make predictions from the start
            model.partial_fit(self.feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)

    # Returns predictions for each model
    def predict(self, s):
        X = self.feature_transformer.transform([s])
        return np.array([model.predict(X)[0] for model in self.models])

    # Performs one step of gradient descent on the given
    # action's model
    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        # Only update the given action's model
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        # Technically, we don't need gradient descent,
        # Because the RBFSampler will predict values to
        # be zero for a while at the beginning. Since the
        # rewards are all negative, this can help us use
        # optimistic initial values instead of gradient descent
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(model, env, eps, gamma):
    observation = env.reset()
    done = False
    total_reward = 0
    iters = 0
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        # update the model
        next = model.predict(observation)
        G = reward + gamma * np.max(next[0])
        model.update(prev_observation, action, G)

        total_reward += reward
        iters += 1

    return total_reward

def plot_cost_to_go(env, estimator, num_tiles = 20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_xlabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title('Cost-To-Go Function')
    fig.colorbar(surf)
    plt.show()

def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        # if t < 100: average the last t rewards
        # if t >= 100: average the last 100 rewards
        running_avg[t] = total_rewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title('Running Average')
    plt.show()

def main(show_plots = True):
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant')
    gamma = 0.99

    N = 300
    total_rewards = np.empty(N)
    for n in range(N):
        eps = 1 / np.sqrt(n + 1)
        total_reward = play_one(model, env, eps, gamma)
        total_rewards[n] = total_reward
        if (n + 1) % 100 == 0:
            print('episode:', n, 'total reward:', total_reward)
    print('avg reward for last 100 episodes:', total_rewards[-100:].mean())
    print('total steps:', -total_rewards.sum())

    if show_plots:
        plt.plot(total_rewards)
        plt.title("Rewards")
        plt.show()

        plot_running_avg(total_rewards)

        plot_cost_to_go(env, model)

    env = wrappers.Monitor(env, 'mountain_car_videos')
    play_one(model, env, 0, 0.99)

main()

