from env import EnvSearch
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


env = EnvSearch(100, 4, 10, 5)   # map_size, agent_num, view_range, target_num
env.rand_reset_target_pos()

max_MC_iter = 1000
fig = plt.figure()
gs = GridSpec(1, 3, figure=fig)
ax1 = fig.add_subplot(gs[0:1, 0:1])
ax2 = fig.add_subplot(gs[0:1, 1:2])
ax3 = fig.add_subplot(gs[0:1, 2:3])
last_obs = np.array([])

for MC_iter in range(max_MC_iter):
    if MC_iter % 100 == 0:
        print(MC_iter)
    ax1.imshow(env.get_full_obs())
    ax2.imshow(env.get_current_joint_obs())
    obs = env.get_cumulative_joint_obs(last_obs)
    ax3.imshow(obs)
    last_obs = obs
    agent_act_list = []
    for i in range(env.agent_num):
        agent_act_list.append(random.randint(0, 4))
    env.step(agent_act_list)
    plt.draw()
    plt.pause(0.1)
