import numpy as np
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from collections import deque
from GridWorld import GridWorld
from algorithm import Q_learning, Double_Q_learning


def plot_policy(ax, env, pi):
    '''
    Refer to the documentation and write code to plot the saved policy.
    You can use draw_env() to draw grid as needed, or you can implement your own method.
    '''

    # Map of characters for corresponding actions in env
    arrows = {'UP': r'$\uparrow$', 'DOWN': r'$\downarrow$',
              'LEFT': r'$\leftarrow$', 'RIGHT': r'$\rightarrow$', }

    # Placing the initial state on a grid for illustration
    initials = np.zeros([env.row_max, env.col_max])
    initials[env.row_max - 1, 0] = 1

    # Placing the trap states on a grid for illustration
    traps = np.zeros([env.row_max, env.col_max])
    for t in env.terminal_states:
        if t != (0, env.col_max - 1):
            traps[t] = 2

    # Placing the terminal state on a grid for illustration
    terminals = np.zeros([env.row_max, env.col_max])
    terminals[(0, env.col_max - 1)] = 3

    # Make a discrete color bar with labels
    labels = ['States', 'Initial\nState', 'Trap\nStates', 'Terminal\nState']
    colors = {0: '#F9FFA4', 1: '#B4FF9F', 2: '#FFA1A1', 3: '#FFD59E'}

    cm = ListedColormap([colors[x] for x in colors.keys()])
    norm_bins = np.sort([*colors.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    # Make normalizer and formatter
    norm = BoundaryNorm(norm_bins, len(labels), clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    ax.imshow(initials + traps + terminals, cmap=cm, norm=norm)

    ax.set_xlim((-0.5, env.col_max - 0.5))
    ax.set_ylim((env.row_max - 0.5, -0.5))
    xitems = np.arange(env.col_max)
    yitems = np.arange(env.row_max)
    ax.set_xticks(xitems)
    ax.set_xticklabels([str(i) for i in xitems + 1])
    ax.set_yticks(yitems)
    ax.set_yticklabels([str(i) for i in xitems + 1])

    ax.grid(False)

    for row in range(env.row_max):
        for col in range(env.col_max):
            ax.text(col, row, arrows[env.actions[np.argmax(
                pi[(row, col)])]], ha='center', va='center', fontsize=15)


def play(environment, agent, num_episodes=10000, save_policy_interval=10000, episode_length=1000, train=True, policy=None):
    if policy is None:
        policy = []
    reward_per_episode = []
    returns = deque(maxlen=100)

    for episode in range(num_episodes):
        timestep = 0
        terminal = False
        while timestep < episode_length and terminal != True:
            current_state = environment.agent_location
            action = agent.action(current_state)
            next_state, reward, terminal = environment.make_step(action)
            timestep += 1

            if train:
                agent.update(current_state, next_state, action, reward)

            if terminal:
                episode_return = environment.reset()

        if episode % save_policy_interval == 0 or episode == num_episodes - 1:
            policy.append(agent.get_Q_table().copy())

        returns.append(episode_return)
        reward_per_episode.append(np.mean(returns))

    return reward_per_episode


def draw_env(env, savefig=True):
    plt.figure(figsize=(env.row_max, env.col_max))
    plt.title('Grid World', fontsize=20)

    # Placing the initial state on a grid for illustration
    initials = np.zeros([env.row_max, env.col_max])
    initials[env.row_max - 1, 0] = 1

    # Placing the trap states on a grid for illustration
    traps = np.zeros([env.row_max, env.col_max])
    for t in env.terminal_states:
        if t != (0, env.col_max - 1):
            traps[t] = 2

    # Placing the terminal state on a grid for illustration
    terminals = np.zeros([env.row_max, env.col_max])
    terminals[(0, env.col_max - 1)] = 3

    # Make a discrete color bar with labels
    labels = ['States', 'Initial\nState', 'Trap\nStates', 'Terminal\nState']
    colors = {0: '#F9FFA4', 1: '#B4FF9F', 2: '#FFA1A1', 3: '#FFD59E'}

    cm = ListedColormap([colors[x] for x in colors.keys()])
    norm_bins = np.sort([*colors.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    # Make normalizer and formatter
    norm = BoundaryNorm(norm_bins, len(labels), clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    plt.imshow(initials + traps + terminals, cmap=cm, norm=norm)
    plt.colorbar(format=fmt, ticks=tickz)

    plt.xlim((-0.5, env.col_max - 0.5))
    plt.ylim((env.row_max - 0.5, -0.5))
    plt.yticks(np.linspace(env.row_max - 0.5, -0.5, env.row_max + 1))
    plt.xticks(np.linspace(-0.5, env.col_max - 0.5, env.col_max + 1))
    plt.grid(color='k')

    for loc in env.terminal_states:
        plt.text(loc[1], loc[0], 'X', ha='center', va='center', fontsize=40)
    plt.text(0, env.row_max - 1, 'O', ha='center', va='center', fontsize=40)

    if savefig:
        plt.savefig('./gridworld.png')


def main():
    np.random.seed(5)
    # Set hyperparameters ([!] you can modify for analyzing algorithms you will implement)
    num_episodes = 20000
    save_policy_interval = 10000
    epsilon = 0.2
    alpha = 0.03

    # Create environment
    env = GridWorld()
    draw_env(env)

    # We provide experiment codes for the project 1

    ##################################
    # train an agent via Q-learninng #
    ##################################
    Q_policy = []
    agent_Q_learning = Q_learning(env, epsilon=epsilon, alpha=alpha)
    reward_per_episode1 = play(env, agent_Q_learning, num_episodes=num_episodes,
                               save_policy_interval=save_policy_interval, policy=Q_policy)

    # Draw policy of Q-learninng
    num_plots = len(Q_policy)
    fig, ax = plt.subplots(num_plots, figsize=(num_plots*2, num_plots*4))
    for i, q in enumerate(Q_policy):
        plot_policy(ax[i], env, q)
        _ = ax[i].set_title(
            "Q-learning Greedy policy at {} th episode".format(i*save_policy_interval))
    plt.savefig('Q_learning_policy.png')

    ########################################
    # train an agent via Double Q-learning #
    ########################################
    env.reset()
    Double_Q_policy = []
    agent_Double_Q_learning = Double_Q_learning(
        env, epsilon=epsilon, alpha=alpha)
    reward_per_episode2 = play(env, agent_Double_Q_learning, num_episodes=num_episodes,
                               save_policy_interval=save_policy_interval, policy=Double_Q_policy)

    # Draw policy of Double Q-learning
    num_plots = len(Double_Q_policy)
    fig, ax = plt.subplots(num_plots, figsize=(num_plots*2, num_plots*4))
    for i, q in enumerate(Double_Q_policy):
        plot_policy(ax[i], env, q)
        _ = ax[i].set_title(
            "Double Q-learning Greedy policy at {} th episode".format(i*save_policy_interval), fontsize=12)
    plt.savefig('double_Q_learning_policy.png')

    # Make learning curve
    plt.figure()
    plt.plot(range(1, num_episodes + 1),
             reward_per_episode1, label="Q-learning")
    plt.plot(range(1, num_episodes + 1),
             reward_per_episode2, label="Double-Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Return per Episode")
    plt.legend()
    plt.savefig("learning_curve.png")

    # Make heatmap for max_a Q(s,a) for each state in the GridWorld
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    max_Q_list = [agent_Q_learning.get_max_Q_function(
    ), agent_Double_Q_learning.get_max_Q_function()]
    per_name = ["Q-learning", "Double-Q-learning"]
    state_list = [i for i in range(10)]
    count = 0
    for ax in axes.flat:
        im = ax.imshow(max_Q_list[count], cmap='viridis')
        ax.set_title(per_name[count], size=10)
        ax.set_xticks(range(len(state_list)))
        ax.set_yticks(range(len(state_list)))
        ax.set_xticklabels(state_list)
        ax.set_yticklabels(state_list)
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        fig.colorbar(im, ax=ax, shrink=1)
        count += 1
    fig.savefig("max_Q_for_each_state.png")


if __name__ == "__main__":
    main()
