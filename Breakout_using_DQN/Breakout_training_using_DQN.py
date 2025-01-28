import tensorflow as tf
import gym
import random
from collections import deque
import numpy as np
import cv2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam

env = gym.make("ALE/Breakout-v5", frameskip=4, render_mode="human")
action_dim = env.action_space.n

image_size = (84, 84)
GAMMA = 0.99
BATCH_SIZE = 64
LR = 1e-4
EPSILON_START = 0.5
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TAU = 0.001
MEMORY_CAPACITY = 100000
EPISODES = 100
STACK_SIZE = 4


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def build_model(input_shape, output_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(32, (8, 8), strides=4, activation='relu')(inp)
    x = Conv2D(64, (4, 4), strides=2, activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(output_shape, activation='linear')(x)
    return Model(inputs=inp, outputs=x)


def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, image_size, interpolation=cv2.INTER_AREA)  # Resize
    normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
    return normalized_frame


def stack_frames(stack, frame, is_new_episode):
    frame = preprocess_frame(frame)
    if is_new_episode:
        stack = deque([frame] * STACK_SIZE, maxlen=STACK_SIZE)
    else:
        stack.append(frame)
    stacked_frames = np.stack(stack, axis=-1)  # Stack along the channel dimension
    return stacked_frames, stack


def update_target_network(online_model, target_model, tau):
    for online_var, target_var in zip(online_model.trainable_variables, target_model.trainable_variables):
        target_var.assign(tau * online_var + (1 - tau) * target_var)


input_shape = (84, 84, STACK_SIZE)
main_model = build_model(input_shape, action_dim)
target_model = build_model(input_shape, action_dim)
target_model.set_weights(main_model.get_weights())
optimizer = Adam(learning_rate=LR)

replay_buffer = ReplayBuffer(MEMORY_CAPACITY)


def epsilon_greedy_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        q_values = main_model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values)


@tf.function
def train_step(states, actions, rewards, next_states, dones):
    next_q_values = target_model(next_states)
    max_next_q_values = tf.reduce_max(next_q_values, axis=1)
    target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

    with tf.GradientTape() as tape:
        q_values = main_model(states)
        one_hot_actions = tf.one_hot(actions, action_dim)
        q_action_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
        loss = tf.reduce_mean(tf.square(target_q_values - q_action_values))

    grads = tape.gradient(loss, main_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_model.trainable_variables))


epsilon = EPSILON_START
stack = deque(maxlen=STACK_SIZE)

for episode in range(EPISODES):
    state = env.reset()[0]
    state, stack = stack_frames(stack, state, is_new_episode=True)
    total_reward = 0

    for t in range(10000):
        action = epsilon_greedy_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state, stack = stack_frames(stack, next_state, is_new_episode=False)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(replay_buffer) >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            train_step(*[tf.convert_to_tensor(x) for x in batch])

        update_target_network(main_model, target_model, TAU)

        if done:
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

print("Training complete!")

target_model.save("Breakout_atari_best_model.keras")