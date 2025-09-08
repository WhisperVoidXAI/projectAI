# DQN simplified training for SpaceInvaders (works with gymnasium)
import numpy as np
import gymnasium as gym
import tensorflow as tf
import tf_slim as slim
from collections import deque
import matplotlib.pyplot as plt
import random
import time

tf.compat.v1.disable_eager_execution()

# ---------- Helpers: preprocessing & frame stacking ----------
def preprocess_state(state):
    # state: HxWx3 uint8
    img = state[25:201:2, ::2]            # crop and downsample
    img = img.mean(axis=2).astype(np.float32)  # to gray
    color = np.array([210, 164, 74]).mean()
    img[img == color] = 0
    img = (img - 128.0) / 128.0           # normalize approx to [-1,1]
    return img.reshape(88, 80)            # final shape

def make_initial_stack(frame, stack_size=4):
    stack = deque([np.zeros((88,80), dtype=np.float32) for _ in range(stack_size)], maxlen=stack_size)
    for _ in range(stack_size):
        stack.append(frame)
    return stack

def stack_to_state(stacked_frames):
    return np.stack(stacked_frames, axis=2)  # shape (88,80,4)


# ---------- Q-network (TensorFlow v1 + tf_slim) ----------
def q_network(X, name_scope, n_outputs):
    initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0)
    with tf.compat.v1.variable_scope(name_scope):
        net = X  # expected shape: [None, 88, 80, 4]
        net = slim.conv2d(net, 32, [8,8], stride=4, padding='SAME', weights_initializer=initializer)
        net = slim.conv2d(net, 64, [4,4], stride=2, padding='SAME', weights_initializer=initializer)
        net = slim.conv2d(net, 64, [3,3], stride=1, padding='SAME', weights_initializer=initializer)
        flat = slim.flatten(net)
        fc = slim.fully_connected(flat, 128, weights_initializer=initializer)
        output = slim.fully_connected(fc, n_outputs, activation_fn=None, weights_initializer=initializer)
        vars = {v.name[len(tf.compat.v1.get_variable_scope().name)+1:]: v for v in
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=tf.compat.v1.get_variable_scope().name)}
    return vars, output


# ---------- Hyperparameters ----------
ENV_ID = "ALE/SpaceInvaders-v5"
render = True
num_episodes = 400
batch_size = 32
buffer_len = 20000
learning_rate = 1e-4
discount_factor = 0.99
eps_start = 1.0
eps_end = 0.05
eps_decay_steps = 200000
stack_size = 4
copy_steps = 500
train_every = 4
start_learning = 2000
max_episode_steps = 10000

# ---------- Environment ----------
env = gym.make(ENV_ID, render_mode="human" if render else None)
n_outputs = int(env.action_space.n)
print("عدد الأفعال:", n_outputs)
print("معاني الأفعال:", env.get_action_meanings())

# ---------- TF placeholders & networks ----------
tf.compat.v1.reset_default_graph()
X = tf.compat.v1.placeholder(tf.float32, shape=(None, 88, 80, stack_size), name='X')
X_next = tf.compat.v1.placeholder(tf.float32, shape=(None, 88, 80, stack_size), name='X_next')
actions_ph = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='actions')
rewards_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='rewards')
dones_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='dones')  # 1.0 if done, else 0.0
is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=())

main_vars, main_q_out = q_network(X, 'mainQ', n_outputs)
target_vars, target_q_out = q_network(X_next, 'targetQ', n_outputs)

# compute Q for chosen actions (for training): use main network for Q(s,a)? (we will compute targets with target net)
one_hot_actions = tf.one_hot(actions_ph, n_outputs, dtype=tf.float32)
pred_q = tf.reduce_sum(input_tensor=main_q_out * one_hot_actions, axis=1)  # shape (batch,)

# targets: r + gamma * max_a' Q_target(s', a') * (1 - done)
max_next_q = tf.reduce_max(input_tensor=target_q_out, axis=1)
target_q = rewards_ph + (1.0 - dones_ph) * discount_factor * max_next_q

loss = tf.reduce_mean(input_tensor=tf.square(target_q - pred_q))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# copy target <- main
assign_ops = []
for var_name, main_var in main_vars.items():
    assign_ops.append(tf.compat.v1.assign(target_vars[var_name], main_var))
copy_target_to_main = tf.group(*assign_ops)

# ---------- Replay buffer ----------
exp_buffer = deque(maxlen=buffer_len)

def sample_memories(batch_size):
    idx = np.random.choice(len(exp_buffer), size=batch_size, replace=False)
    batch = [exp_buffer[i] for i in idx]
    states = np.array([b[0] for b in batch], dtype=np.float32)
    actions = np.array([b[1] for b in batch], dtype=np.int32)
    rewards = np.array([b[2] for b in batch], dtype=np.float32)
    next_states = np.array([b[3] for b in batch], dtype=np.float32)
    dones = np.array([b[4] for b in batch], dtype=np.float32)
    return states, actions, rewards, next_states, dones

# ---------- Training loop ----------
saver = tf.compat.v1.train.Saver()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(copy_target_to_main)

global_step = 0
epsilon = eps_start

for ep in range(1, num_episodes+1):
    obs, info = env.reset()
    frame = preprocess_state(obs)
    stacked = make_initial_stack(frame, stack_size)
    state = stack_to_state(stacked)   # shape (88,80,4)
    episode_reward = 0.0
    done = False
    steps = 0

    while not done and steps < max_episode_steps:
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * (global_step / eps_decay_steps))
        if np.random.rand() < epsilon or global_step < start_learning:
            action = env.action_space.sample()
        else:
            q_vals = sess.run(main_q_out, feed_dict={X: state.reshape(1,88,80,stack_size), is_training_ph: False})
            action = np.argmax(q_vals[0])
        next_obs, reward, terminated, truncated, info = env.step(action)
        done_flag = float(1.0 if (terminated or truncated) else 0.0)
        next_frame = preprocess_state(next_obs)
        stacked.append(next_frame)
        next_state = stack_to_state(stacked)
        exp_buffer.append((state, action, reward, next_state, done_flag))
        state = next_state
        episode_reward += reward
        steps += 1
        global_step += 1
        if (global_step > start_learning) and (len(exp_buffer) >= batch_size) and (global_step % train_every == 0):
            states_b, actions_b, rewards_b, next_states_b, dones_b = sample_memories(batch_size)
            feed = {
                X: states_b,
                X_next: next_states_b,
                actions_ph: actions_b,
                rewards_ph: rewards_b,
                dones_ph: dones_b,
                is_training_ph: True
            }
            _, loss_val = sess.run([training_op, loss], feed_dict=feed)
        if global_step % copy_steps == 0:
            sess.run(copy_target_to_main)
        if terminated or truncated:
            done = True
    print(f"Episode {ep:03d} | Steps {steps} | Reward {episode_reward:.1f} | Eps {epsilon:.3f} | GlobalStep {global_step}")
    if ep % 50 == 0:
        saver.save(sess, "./dqn_spaceinvaders.ckpt")
print("Training finished.")
env.close()
sess.close()