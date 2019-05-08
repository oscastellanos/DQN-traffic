import random
import gym
import gym_traffic
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from time import time

from scores.score_logger import ScoreLogger

ENV_NAME = "traffic-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

OBS_SPACE = 28

class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.tensorboard = TensorBoard(log_dir="./logs/")
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(28, input_shape=(28,), activation="relu"))
        self.model.add(Dense(28, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            #print("STATE")
            #print(state)
            #print(np.asarray(state).shape)
            #print("STATE_NEXT")
            #print(state_next)
            #print("STATE-RESHAPED")
            state_reshaped = np.reshape(state[0], (1, 28))
            #print(state_reshaped)
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state_reshaped)
            q_values[0][action] = q_update
            self.model.fit(state_reshaped, q_values, verbose=0, callbacks=[self.tensorboard])
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def phase(env, action, frames):
    t_step = 0
    for _ in range(frames):
        next_state, reward_current, done, _ = env.step(action)
        t_step += 1
    return next_state, reward_current, done, _, t_step

def traffic():
    env = gym.make(ENV_NAME)
    #env = gym.wrappers.Monitor(env, "dqn")
    env.seed(1)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while run < 1:
        run += 1
        step = 0
        state = env._reset()
        obs0, reward_previous, don, signal = state

        reward_current = 0
        total_reward = reward_previous - reward_current
        print("STATUS")
        print(signal)
        if (signal == 0):
            status = 0
        elif (signal == 1):
            status = 1
        next_state = obs0

        while step < 1000:
            #env.render()
            step += 1
            action = dqn_solver.act(state)
            #print(next_state)
            #action = env.action_space.sample()
            #obs1, reward_previous, done, _ = env.step(action)
            
            
            if (status == 0 and action == 0):
                print("Status is: 0. Action is 0.")
                status = 0
                next_state, reward_current, done, _, t_step= phase(env, 0, 15)
                
                step += t_step
                
            elif (status == 0 and action == 1):
                print("Status is 0. Action is now 1. Switching to Status 1.")
                phase(env, 2, 25)
                #print("Action is 1. Status is 0. Lights are H-Y, V-R -> H-R, V-G")
                status = 1
                next_state, reward_current, done, _, t_step = phase(env, 1, 45)
                step += t_step
                
            
            elif (status == 1 and action == 1):
                print("Status is 1. Action is 1.")
                status = 1
                next_state, reward_current, done, _, t_step = phase(env, 1, 15)
                step += t_step
                    
            
            elif (status == 1 and action == 0):
                print("Status is 1. Action is now 0. Switching to Status 0.")
                phase(env, 4, 25)
                status = 0
                next_state, reward_current, done, _, t_step = phase(env, 0, 45)
                step += t_step
                
            
            total_reward = reward_previous - reward_current
            state_next = np.reshape(next_state, [1, OBS_SPACE]) 
            dqn_solver.remember(state, action, total_reward, state_next, done)
            state = state_next
            score_logger.add_score(step, run)

            print(step)
            if done:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                #score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()

    print("Episode done in %d steps, total reward %.2f" % (step, total_reward))
    env.close()

if __name__ == "__main__":
    traffic()
