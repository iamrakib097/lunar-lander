import time
from collections import deque, namedtuple
import numpy as np
import random
from utils import *
from agent import *


random.seed(SEED)
start = time.time()
num_episodes = 2000
max_num_timesteps = 1000
total_point_history = []
num_p_av = 100
epsilon = 1.0

memory_buffer = deque(maxlen=MEMORY_SIZE)
target_q_network.set_weights(q_network.get_weights())

for i in range(num_episodes):
    state = env.reset()
    total_points = 0

    for t in range(max_num_timesteps):
        state_qn = np.expand_dims(state, axis=0)
        q_values = q_network(state_qn)
        action = get_action(q_values, epsilon)

        next_state, reward, done, _ = env.step(action)

        memory_buffer.append(experience(state, action, reward, next_state, done))

        update = check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

        if update:
            experiences = get_experiences(memory_buffer)
            agent_learn(experiences, GAMMA)

        state = next_state.copy()
        total_points += reward

        if done:
            break

    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    epsilon = get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")
        filename = "./lunar_lander.mp4"
        create_video(filename, env, q_network)
        embed_mp4(filename)

tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
plot_history(total_point_history)
import logging
logging.getLogger().setLevel(logging.ERROR)

