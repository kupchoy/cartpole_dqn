import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#%%
env = gym.make('CartPole-v0')
#%%
state_size = env.observation_space.shape[0]
state_size
#%%
action_size = env.action_space.n
action_size
#%%
batch_size = 32
n_episodes = 1001
#%%
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done, in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

#%%
# initialize gym environment and the agent
env = gym.make('CartPole-v0')
agent = DQNAgent(state_size, action_size)

# Iterate the game
for e in range(n_episodes):

    # reset state in the beginning of each game
    state = env.reset()
    state = np.reshape(state, [1, 4])

    # time_t represents each frame of the game
    # Our goal is to keep the pole upright as long as possible until score of 500
    # the more time_t the more score
    for time_t in range(500):
        # turn this on if you want to render
        # env.render()

        # Decide action
        action = agent.act(state)

        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward, done, _ = env.step(action)


        reward = reward if not done else -10
        #
        next_state = np.reshape(next_state, [1, 4])

        # memorize the previous state, action, reward, and done
        agent.memorize(state, action, reward, next_state, done)

        # make next_state the new current state for the next frame.
        state = next_state

        # done becomes True when the game ends
        # ex) The agent drops the pole
        if done:
            # print the score and break out of the loop
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, n_episodes, time_t, agent.epsilon))

            break
    if len(agent.memory) > batch_size:
        # train the agent with the experience of the episode
        agent.replay(batch_size)

    if e % 50 == 0:
        agent.save("weights_" + '{:04d}'.format(e) + '.hdf5')

 #%%
model = Sequential()
model.add(Dense(24, input_dim=4, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=0.001))
#%%
env = gym.make('CartPole-v1')
 #%%
scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(500):
        # env.render()

        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            state = model.predict(np.reshape(prev_obs, [1, 4]))
            action = np.argmax(state[0])

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done: break

    scores.append(score)

print('Average Score for model:',sum(scores)/len(scores))
print('choice 1: {0:1.2f}%  choice 0: {1:1.2f}%'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))

"""
Average Score for model: 400.4
choice 1: 50%  choice 0: 50%

"""
#%%
