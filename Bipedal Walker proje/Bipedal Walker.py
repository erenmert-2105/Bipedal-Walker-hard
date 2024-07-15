import gym
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from collections import deque
import random

# Set TensorFlow logging level to suppress verbose output
tf.get_logger().setLevel('ERROR')

# Ortamı oluşturun
env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")

# Actor ve Critic Ağlarını Tanımlayın
def create_actor(state_dim, action_dim, max_action):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(400, activation="relu")(inputs)
    x = layers.Dense(300, activation="relu")(x)
    outputs = layers.Dense(action_dim, activation="tanh")(x)
    outputs = outputs * max_action
    return models.Model(inputs, outputs)

def create_critic(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    concat = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(400, activation="relu")(concat)
    x = layers.Dense(300, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    return models.Model([state_input, action_input], outputs)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

actor = create_actor(state_dim, action_dim, max_action)
critic = create_critic(state_dim, action_dim)

actor_target = create_actor(state_dim, action_dim, max_action)
critic_target = create_critic(state_dim, action_dim)

actor_optimizer = optimizers.Adam(learning_rate=0.0001)
critic_optimizer = optimizers.Adam(learning_rate=0.001)

critic.compile(optimizer=critic_optimizer, loss='mse')

# Yardımcı Fonksiyonlar
def update_target_weights(model, target_model, tau):
    model_weights = model.get_weights()
    target_weights = target_model.get_weights()
    new_weights = []
    for model_weight, target_weight in zip(model_weights, target_weights):
        new_weight = tau * model_weight + (1 - tau) * target_weight
        new_weights.append(new_weight)
    target_model.set_weights(new_weights)

def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    state = np.array(state)
    state = np.expand_dims(state, axis=0)
    return actor.predict(state, verbose=0)[0]

def train():
    minibatch = random.sample(replay_buffer, batch_size)
    states = np.array([np.asarray(transition[0]) for transition in minibatch])
    actions = np.array([np.asarray(transition[1]) for transition in minibatch])
    rewards = np.array([transition[2] for transition in minibatch])
    next_states = np.array([np.asarray(transition[3]) for transition in minibatch])
    dones = np.array([transition[4] for transition in minibatch])

    target_qs = critic_target.predict([next_states, actor_target.predict(next_states, verbose=0)], verbose=0)
    target_qs = rewards + gamma * (1 - dones) * target_qs

    critic_loss = critic.train_on_batch([states, actions], target_qs)

    with tf.GradientTape() as tape:
        actions_pred = actor(states, training=True)
        critic_value = critic([states, actions_pred])
        actor_loss = -tf.reduce_mean(critic_value)

    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    update_target_weights(actor, actor_target, tau)
    update_target_weights(critic, critic_target, tau)

    return actor_loss.numpy(), critic_loss

# Hiperparametreler
gamma = 0.99
tau = 0.005
replay_buffer = deque(maxlen=10000)
batch_size = 32
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

# Loss ve reward değerlerini kaydetmek için numpy array'ler
actor_losses = np.array([])
critic_losses = np.array([])
total_rewards = np.array([])

# Eğitim Döngüsü
num_episodes = 10000
rewards_epsilon = np.zeros((num_episodes, 2))  # İki sütunlu numpy array

for episode in range(num_episodes):
    state = np.array(env.reset()[0])
    episode_reward = 0
    value = 2000
    if episode <= 1000:
        value = 200
        if episode >= 5000:
            value = 10000
    for t in range(value):
        action = select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.asarray(next_state)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward

        if done:
            break

    if len(replay_buffer) > batch_size:
        actor_loss, critic_loss = train()
        actor_losses = np.append(actor_losses, actor_loss)
        critic_losses = np.append(critic_losses, critic_loss)
    else:
        actor_loss, critic_loss = 0, 0  # Default values if no training is performed

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    total_rewards = np.append(total_rewards, episode_reward)
    rewards_epsilon[episode] = [epsilon, episode_reward]
    print(f"Episode: {episode}, Reward mean: {episode_reward / episode:.4f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")


# Loss ve reward değerlerini kaydetme
np.savetxt("actor_losses.txt", actor_losses)
np.savetxt("critic_losses.txt", critic_losses)
np.savetxt("total_rewards.txt", total_rewards)
np.savetxt("rewards_epsilon.txt", rewards_epsilon)
