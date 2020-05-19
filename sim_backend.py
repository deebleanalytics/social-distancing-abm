import numpy as np
from global_variables import G

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
import seaborn as sns
from tqdm import tqdm


# Use sns styling and TkAgg backend
sns.set_style("white")
matplotlib.use("TkAgg")


class Agents:
    def __init__(self):
        self.agents_count = 0
        # Goals are left-to-right as 0 and right-to-left as 1
        self.goals = []

        # Position, velocity, and goal direction are r, v, and e respectively
        self.r = []
        self.v = []
        self.e = []

        # v_pref is the velocity which the agent tends towards without external influence
        self.v_pref = []
        # v_max is the maximum possible velocity of the agent
        self.v_max = []

        # socially_distant is 1 if no agent within minimum distance or 0 otherwise
        self.socially_distant = []
        # min_distance is the distance to the next closest agent
        self.min_distance = []

    def add_agents(self, r=(.0, .0), goal=0, v=(.0, .0), v_pref=1, v_max=1.2):
        self.agents_count += 1
        self.goals = np.append(self.goals, goal)

        self.r = np.append(self.r, r).reshape(self.agents_count, 2)
        self.v = np.append(self.v, v).reshape(self.agents_count, 2)

        # Below currently doesn't handle multiple inputs
        self.e = np.append(self.e, self.e_lookup(goal)).reshape(self.agents_count, 2)
        self.v_pref = np.append(self.v_pref, v_pref).reshape(self.agents_count, 1)
        self.v_max = np.append(self.v_max, v_max).reshape(self.agents_count, 1)
        self.socially_distant = np.append(self.socially_distant, 1)

    def delete_agents(self, index):
        self.agents_count -= 1
        self.goals = np.delete(self.goals, index, 0)

        self.r = np.delete(self.r, index, 0)
        self.v = np.delete(self.v, index, 0)
        self.e = np.delete(self.e, index, 0)
        self.v_pref = np.delete(self.v_pref, index, 0)
        self.v_max = np.delete(self.v_max, index, 0)
        self.socially_distant = np.delete(self.socially_distant, index, 0)

    @staticmethod
    def e_lookup(goal):  # This method is set to be replaced by a Dijkstra method for complex goals
        if goal == 0:
            return 1.0, .0
        elif goal == 1:
            return -1.0, .0
        else:
            pass

    # Notation as per Social Forces paper
    def w(self, f):
        e_dot_f = np.sum(self.e[:, np.newaxis] * f, axis=-1)
        f_cos_phi = np.linalg.norm(f, axis=-1) * np.cos(G.phi / 2.0 / 180.0 * np.pi)

        w_temp = (e_dot_f > f_cos_phi).astype(float)
        w_temp[w_temp == 0] = G.c
        return w_temp

    # Notation as per Social Forces paper
    def g(self, w_a):
        g_temp = self.v_max / np.linalg.norm(w_a, axis=0)
        return np.minimum(g_temp, np.ones(g_temp.shape))

    # Notation as per Social Forces paper
    def calc_f_a(self):
        # noinspection PyTypeChecker
        return (self.v_pref * self.e - self.v) / G.tau

    # Custom function giving a force of G.SOCIAL_DISTANCE_FORCE to separate agents within
    #   the minimum social distance
    def calc_f_sd(self):
        """Social distancing of a scaled force if within minimum social distance"""
        r_ab = np.expand_dims(self.r, 0) - np.expand_dims(self.r, 1)
        r_ab_norm = np.linalg.norm(r_ab, axis=-1)
        np.fill_diagonal(r_ab_norm, np.inf)
        mask = np.where(r_ab_norm < G.SOCIAL_DISTANCE, 1 / r_ab_norm, 0)
        return mask[:, :, np.newaxis] * r_ab * G.SOCIAL_DISTANCE_FORCE

    # Notation as per Social Forces paper
    def b(self, r_ab):
        s = np.linalg.norm(self.v, axis=-1) * G.DT
        b_temp = (np.linalg.norm(r_ab, axis=-1) +
                  np.linalg.norm(r_ab - s[np.newaxis, :, np.newaxis] * self.e[np.newaxis, :], axis=-1)
                  ) ** 2 - (s[np.newaxis, :]) ** 2
        np.fill_diagonal(b_temp, .0)
        return 0.5 * np.sqrt(b_temp)

    # Notation as per Social Forces paper
    def v_ab(self, r_ab):
        return G.V_ab0 * np.exp(-self.b(r_ab) / G.SIGMA)

    # Notation as per Social Forces paper
    def calc_f_ab(self):
        r_ab = np.expand_dims(self.r, 0) - np.expand_dims(self.r, 1)

        dx = np.array([[[G.DT_FD, 0.0]]])
        dy = np.array([[[0.0, G.DT_FD]]])

        v = self.v_ab(r_ab)
        dv_dx = (self.v_ab(r_ab + dx) - v) / G.DT_FD
        dv_dy = (self.v_ab(r_ab + dy) - v) / G.DT_FD

        np.fill_diagonal(dv_dx, 0.0)
        np.fill_diagonal(dv_dy, 0.0)

        return -1 * np.stack((dv_dx, dv_dy), axis=-1)

    # Notation as per Social Forces paper
    @staticmethod
    def r_aB(r):
        # Very basic return of distance from upper and lower limit of environment
        return np.transpose([r[:, 1], G.ENVIRONMENT_SIZE[1] - r[:, 1]])

    # Notation as per Social Forces paper
    def u_aB(self, r):
        return G.U_aB0 * np.exp(-self.r_aB(r) / G.R)

    # Notation as per Social Forces paper
    def calc_f_aB(self):
        dy = np.array([[0.0, G.DT_FD]])

        u = self.u_aB(self.r)

        dv_dy = (self.u_aB(self.r + dy) - u) / G.DT_FD
        dv_dx = np.zeros(dv_dy.shape)

        return -1 * np.stack((dv_dx, dv_dy), axis=-1)

    # Notation as per Social Forces paper
    def calc_f(self):
        f_a = self.calc_f_a()
        f_sd = self.calc_f_sd()
        f_ab = self.calc_f_ab()
        f_aB = self.calc_f_aB()

        w_f_ab = self.w(f_ab)
        f = f_a + \
            np.sum(f_sd, axis=0) + \
            np.sum(f_ab * w_f_ab[:, :, np.newaxis], axis=0) + \
            np.sum(f_aB, axis=1)
        return f


# Animator class used for processing time steps and plotting
class Animator:
    def __init__(self, animator_agents, file_name=None):
        self.initial_agents = animator_agents.agents_count
        self.agents = animator_agents
        self.file_name = file_name
        self.fig, self.ax = plt.subplots(2)
        self.frames = int(G.SIMULATION_DURATION * G.FPS / G.ANIMATION_PLAY_SPEED)
        self.p_bar = tqdm(total=self.frames)

        self.scat, self.time_text, self.average_distance_to_nearest = None, None, None
        self.people, self.average_distance = [], []

    def setup_plot(self):
        self.ax[0].set_aspect('equal')
        self.ax[0].set(xlim=(0, G.ENVIRONMENT_SIZE[0]),
                       ylim=(0, G.ENVIRONMENT_SIZE[1]))
        self.ax[1].set(xlim=(0, 50),
                       ylim=(0, 70))
        self.ax[1].set(xlabel="Number of agents",
                       ylabel="Average distance to nearest neighbour")
        self.ax[1].grid(True)

        self.time_text = self.ax[0].text(0.02, 0.9, '', transform=self.ax[0].transAxes)
        self.scat = self.ax[0].scatter(self.agents.r[:, 0],
                                       self.agents.r[:, 1],
                                       c=G.COLOUR_PALETTE(self.agents.socially_distant),
                                       s=8)

        self.average_distance_to_nearest = self.ax[1].plot([0], [0], lw=2)

        return self.scat, self.time_text, self.average_distance_to_nearest

    def process_frame(self, frame):
        # Agent positions are calculated by DT time-steps and a frame is created when this exceeds
        #   the next frame time from FPS and PLAY_SPEED
        while G.current_simulation_time < frame / G.FPS * G.ANIMATION_PLAY_SPEED:
            self.process_agents()
            G.current_simulation_time += G.DT

        # Update progress bar
        self.p_bar.update(1)

        # Plot the positions of the agents and colour by if they are within social distance
        self.scat.set_offsets(self.agents.r)
        self.scat.set_color(G.COLOUR_PALETTE(self.agents.socially_distant))

        # If there are more than one agents, plot the average distance against number of agents.
        #   Note that alpha is used and the scatter points are overlayed ontop of each other.
        if len(self.agents.min_distance) > 0:
            self.people.append(self.agents.agents_count)
            self.average_distance.append(np.nanmean(self.agents.min_distance))

            self.average_distance_to_nearest = self.ax[1].scatter(self.people,
                                                                  self.average_distance,
                                                                  c='blue',
                                                                  alpha=0.008,
                                                                  marker="s",
                                                                  s=10)

        # Update the time
        self.time_text.set_text("Time: " + str(round(G.current_simulation_time)) + '/' + str(G.SIMULATION_DURATION))

        return self.scat, self.time_text, self.average_distance_to_nearest

    def process_agents(self):
        # Delete agents that have reached the far end of the corridor
        delete_list = []
        for i in range(self.agents.agents_count):
            if (self.agents.goals[i] == 0) & (self.agents.r[i, 0] > G.ENVIRONMENT_SIZE[0]):
                delete_list.append(i)
            elif (self.agents.goals[i] == 1) & (self.agents.r[i, 0] < 0):
                delete_list.append(i)
        for i in delete_list:
            self.agents.delete_agents(i)

        # Process time-step for agents present and move them forward
        if self.agents.agents_count > 0:
            f = self.agents.calc_f()
            w_a = self.agents.v + f * G.DT
            self.agents.v = (w_a * self.agents.g(w_a))
            self.agents.r += self.agents.v * G.DT

            # Add the distance to their closest neighbour and flag if not socially distant
            r_ab = np.expand_dims(self.agents.r, 0) - np.expand_dims(self.agents.r, 1)
            distance = np.linalg.norm(r_ab, axis=-1)
            np.fill_diagonal(distance, np.inf)
            self.agents.min_distance = np.min(distance, axis=-1)
            self.agents.socially_distant = self.agents.min_distance > G.SOCIAL_DISTANCE

        # Add agents with a poisson distribution (randomly arriving)
        # Lineally increase the frequency of arrivals
        G.poisson_lambda = G.poisson_lambda_range[0] + G.current_simulation_time * (
                G.poisson_lambda_range[1] - G.poisson_lambda_range[0]) / G.SIMULATION_DURATION

        # Get events in time period and add
        arrivals = 0
        while G.current_simulation_time > G.next_arrival_time:
            arrivals += 1
            G.next_arrival_time += -np.log(1.0 - np.random.random()) / G.poisson_lambda

        # For each arrival in the time period, assign a side and starting position randomly
        for i in range(arrivals):
            rand = np.random.choice([0, 1])
            y = np.random.choice(np.linspace(0 + G.SPAWN_BUFFER, G.ENVIRONMENT_SIZE[1] - G.SPAWN_BUFFER))

            if rand == 0:
                self.agents.add_agents(r=(-G.SPAWN_BUFFER, y),
                                       v=(1, 0),
                                       goal=0)
            else:
                self.agents.add_agents(r=(G.ENVIRONMENT_SIZE[0] + G.SPAWN_BUFFER, y),
                                       v=(-1, 0),
                                       goal=1)

    def run(self):
        # Run the animation
        animation_plot = animation.FuncAnimation(self.fig,
                                                 self.process_frame,
                                                 frames=self.frames,
                                                 init_func=self.setup_plot,
                                                 interval=1000 / G.FPS,
                                                 blit=False)
        # Save if file_name given
        if isinstance(self.file_name, str):
            ff_writer = animation.FFMpegWriter(fps=G.FPS)
            animation_plot.save(self.file_name, writer=ff_writer)
        else:
            plt.show()

        # Close the progress bar
        self.p_bar.close()