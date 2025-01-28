import gym
from collections import deque
import numpy as np
import cv2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten


env = gym.make('ALE/Breakout-v5', render_mode="human")
image_size = (84, 84)
STACK_SIZE = 8

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, image_size, interpolation=cv2.INTER_AREA)
    normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
    return normalized_frame
def stack_frames(stack, frame, is_new_episode):
    frame = preprocess_frame(frame)
    if is_new_episode:
        stack = deque([frame] * STACK_SIZE, maxlen=STACK_SIZE)
    else:
        stack.append(frame)
    stacked_frames = np.stack(stack, axis=-1)
    return stacked_frames, stack
state = env.reset()[0]
done = False
action = 1
i = 0
model = load_model("Breakout_atari_model_with_8_stack.keras")
is_new_episode = True
stack = deque(maxlen=STACK_SIZE)
stacked_frames, stack = stack_frames(stack, state, is_new_episode)
total_re = 0
while not done:
    is_new_episode = False
    stacked_frames, stack = stack_frames(stack, state, is_new_episode)
    state = np.expand_dims(stacked_frames, axis=0)
    action = np.argmax(model(state))
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
    total_re += reward
    env.render()
print(f"Total Reward: {total_re}")
env.close()
