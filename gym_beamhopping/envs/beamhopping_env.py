"""
Dynamic beamhopping and resource allocation design for LEO satellite systems, with its parametrized action space,
Discrete action: beam pattern
Continuous action/parameter: bandwidth and power allocation for illuminated cells
Observation/state space: traffic and satellite position in each time slot
"""
from abc import ABC

import numpy as np
import math
import gym
from gym import spaces, error
from gym.utils import seeding
import random
from scipy.stats import poisson
from .config import n_cell, n_beam, p_total, b_total, max_lambda, min_lambda, max_gain_tx, gain_rx, noise_power, \
    t_ttl, TS_per_episode, TS_duration, weight_factor, r_E, d_S, v_S, location_factor, r_C, location_cells, S0_proj, \
    distance_TS, f_c, c, lambda_c, max_throughput, max_delay_fair

# discrete action: point beam pattern
DISCRETE_ACTION = []
for i in range(0, n_cell - n_beam):
    for j in range(i + 1, n_cell - n_beam + 1):
        for m in range(j + 1, n_cell - n_beam + 2):
            for n in range(m + 1, n_cell - n_beam + 3):
                DISCRETE_ACTION.append([i, j, m, n])

DISCRETE_ACTION_SPACE = {i: j for i, j in enumerate(DISCRETE_ACTION)}
num_actions = len(DISCRETE_ACTION_SPACE)  # number of discrete actions, C_19^4

# range for continuous action/parameters, {p0, p1, p2, p3, B0, E0, B1, E1, B2, E2, B3, E3}
PARAMETERS_MIN = []
PARAMETERS_MAX = []
for i in range(num_actions):
    PARAMETERS_MIN.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
for j in range(num_actions):
    PARAMETERS_MAX.append(
        np.array([p_total / n_beam, p_total / n_beam, p_total / n_beam, p_total / n_beam, b_total, b_total,
                  b_total, b_total, b_total, b_total, b_total, b_total]))


def norm(vec2d):
    # from numpy.linalg import norm
    # faster to use custom norm because we know the vectors are always 2D
    assert len(vec2d) == 2
    return math.sqrt(vec2d[0] * vec2d[0] + vec2d[1] * vec2d[1])


def band_func(begin, end):
    # band satisfies begin <= end, or = 0
    if begin <= end:
        return end - begin
    else:
        return 0


def overlapping_factor(begin1, end1, begin2, end2):
    # calculate overlapping factor for user1 and user2
    if band_func(begin1, end1) == 0 or band_func(begin2, end2) == 0:
        return 0  # bandwidth=0
    else:
        bandwidth = band_func(begin1, end1)
        if end1 > begin2 and end2 > begin1:
            factor = (min(end1, end2) - max(begin1, begin2)) / bandwidth
            return factor
        else:
            return 0  # non-overlapping


class BeamHoppingEnv(gym.Env):
    # metadata = {'render.modes': ['human', 'rgb_array']}
    metadata = {'render.modes': ['human']}  # cannot use rgb_array at the moment due to frame skip between actions

    def __init__(self):
        """ The entities are set up and added to a space. """

        self.np_random = None

        self.states = []
        self.render_states = []

        self.time = 0
        self.max_time = TS_per_episode

        self.arrival_rate_max = random.randrange(min_lambda, max_lambda)  # varies at every TS_per_episode
        self.arrival_rate_cell = []  # arrival rate for each cell
        for kk in range(n_cell):
            self.arrival_rate_cell.append(self.arrival_rate_max * location_factor[kk])

        self.action_space = spaces.Tuple((
            spaces.Discrete(num_actions),  # actions
            spaces.Tuple(  # parameters
                tuple(spaces.Box(PARAMETERS_MIN[ii], PARAMETERS_MAX[ii], dtype=np.float32) for ii in range(num_actions))
            )
        ))
        self.observation_space = spaces.Tuple((
            # spaces.Box(low=0., high=1., shape=self.get_state().shape, dtype=np.float32),  # scaled states
            # spaces.Box(low=LOW_VECTOR, high=HIGH_VECTOR, dtype=np.float32),  # unscaled states
            # spaces.Discrete(200),  # internal time steps (200 limit is an estimate)
            spaces.Discrete(2 + n_cell * t_ttl),  # size of observation space: LEO projection position+traffic for cells
        ))

        self.seed()

    def step(self, action):
        """
        Take a full, stabilised update.

        Parameters
        ----------
        action (ndarray) :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
            reward (float) :
            terminal (bool) :
            info (dict) :
        """
        act_index = action[0]
        act = DISCRETE_ACTION_SPACE[act_index]  # discrete actions=picked cells
        param = action[1][act_index]
        param = np.clip(param, PARAMETERS_MIN[act_index], PARAMETERS_MAX[act_index])  # continuous parameters/actions
        observation = self.states[-1]  # current state

        self.time += 1
        if self.time == self.max_time:
            end_episode = True
            # arrival_rate_max changes when time == max_time (TS_per_episode)
            self.arrival_rate_max = random.randrange(min_lambda, max_lambda)  # varies at every TS_per_episode
            self.arrival_rate_cell = []  # arrival rate for each cell
            for kk in range(n_cell):
                self.arrival_rate_cell.append(self.arrival_rate_max * location_factor[kk])
            self.time = 0

            return {}, {}, end_episode, {}
        end_episode = False

        # Return new_state, reward, end_episode
        position_observation = observation[:2]
        traffic_observation = observation[2:]
        traffic_observation = traffic_observation.reshape([n_cell, t_ttl])
        position_observation_ = position_observation.copy()
        traffic_observation_ = traffic_observation.copy()

        # Satellite projection point moves at the projected orbit with distance_TS
        position_observation_[0] = position_observation_[0] + 0.5 * math.sqrt(3) * distance_TS
        position_observation_[1] = position_observation_[1] + 0.5 * distance_TS

        # provide service for picked cells, continuous parameters: {p0, p1, p2, p3, B0, E0, B1, E1, B2, E2, B3, E3}
        picked_cells = act
        resource_picked_cells = np.zeros((n_beam, 3))
        for nn in range(n_beam):
            resource_picked_cells[nn, 0] = param[nn]  # power
            resource_picked_cells[nn, 1] = param[2 * nn + n_beam]  # begin bandwidth
            resource_picked_cells[nn, 2] = param[2 * nn + n_beam + 1]  # end bandwidth

        # Initializations
        channel_cap = np.zeros(n_beam)  # channel capacity for picked cells (=n_beam)
        SINR = np.zeros(n_beam)  # SINR for picked cells
        distance_LoS = np.zeros(n_beam)  # LoS distance between satellite and picked cells
        path_loss = np.zeros(n_beam)  # path loss for picked cells
        h = np.zeros(n_beam)  # channel for picked cells

        # Calculate channel coefficient for picked cells
        for nn in range(n_beam):
            location_picked_cell = location_cells[picked_cells[nn]]
            distance_projection = norm(position_observation_ - location_picked_cell)
            distance_LoS[nn] = math.sqrt(distance_projection * distance_projection + d_S * d_S)
            path_loss[nn] = pow(4 * math.pi * distance_LoS[nn] / lambda_c, 2)
            path_loss[nn] = 10 * math.log10(path_loss[nn])  # dB
            h[nn] = max_gain_tx + gain_rx - path_loss[nn]  # dB
            h[nn] = 10 ** (h[nn] / 10)

        # Calculate channel capacity for picked cells
        for nn in range(n_beam):
            power = resource_picked_cells[nn, 0]
            bandwidth = band_func(resource_picked_cells[nn, 1], resource_picked_cells[nn, 2])
            begin1 = resource_picked_cells[nn, 1]
            end1 = resource_picked_cells[nn, 2]
            interference = 0
            for mm in range(n_beam):  # calculate interference
                if mm != nn:
                    power_interference = resource_picked_cells[mm, 0]
                    begin2 = resource_picked_cells[mm, 1]
                    end2 = resource_picked_cells[mm, 2]
                    overlap_fac = overlapping_factor(begin1, end1, begin2, end2)
                    interference += power_interference * h[mm] * overlap_fac
            SINR[nn] = (power * h[nn]) / (noise_power * bandwidth + interference)
            channel_cap[nn] = bandwidth * math.log2(1 + SINR[nn]) * TS_duration  # * TS_duration (bits)

            # Process data in the queue
            rest_capacity = channel_cap[nn]
            for t in range(t_ttl):
                if traffic_observation_[nn, t_ttl - t - 1] <= rest_capacity:
                    traffic_observation_[nn, t_ttl - t - 1] = 0
                    rest_capacity -= traffic_observation_[nn, t_ttl - t - 1]
                else:
                    traffic_observation_[nn, t_ttl - t - 1] -= rest_capacity
                    rest_capacity = 0
                if rest_capacity == 0:
                    break

        # Calculate reward
        throughput = np.zeros(n_beam)  # throughput for each picked cell
        tau = np.zeros(n_cell)
        phi = np.zeros(t_ttl)
        phi_temp = np.zeros(t_ttl)

        for nn in range(n_beam):
            traffic_picked_cell = sum(traffic_observation[picked_cells[nn], :])
            throughput[nn] = min((traffic_picked_cell, channel_cap[nn]))

        for kk in range(n_cell):
            for tt in range(t_ttl):
                phi[tt] = traffic_observation_[kk,tt]
                phi_temp[tt] = (tt + 1) * phi[tt]
            tau[kk] = sum(phi_temp) / sum(phi)
            if math.isnan(tau[kk]):
                tau[kk] = 0
        delay_fair = max(tau) - min(tau)
        reward = weight_factor * sum(throughput) / max_throughput - (1-weight_factor)*delay_fair/max_delay_fair

        # new arrival data for next traffic_observation
        traffic_observation_[:, 1:] = traffic_observation_[:, 0:(t_ttl - 1)]
        for kk in range(n_cell):
            traffic_observation_[kk, 0] = poisson.rvs(self.arrival_rate_cell[kk]) * 1e6 * TS_duration

        traffic_observation_ = traffic_observation_.reshape(n_cell * t_ttl)
        position_observation_ = np.array(position_observation_)
        observation_ = np.concatenate((position_observation_, traffic_observation_))
        self.states.append(observation_)
        self.render_states.append(self.states[-1])

        return observation_, reward, end_episode, {}

    # build initial states
    def reset(self):
        # TODO: implement reset for each entity to avoid creating new objects and reduce duplicate code
        self.states = []
        self.render_states = []

        self.time = 0

        self.arrival_rate_max = random.randrange(min_lambda, max_lambda)  # varies at every TS_per_episode
        self.arrival_rate_cell = []  # arrival rate for each cell
        for kk in range(n_cell):
            self.arrival_rate_cell.append(self.arrival_rate_max * location_factor[kk])

        traffic_observation = np.zeros([n_cell, t_ttl])  # traffic of each cell initially
        for kk in range(n_cell):
            traffic_observation[kk, 0] = poisson.rvs(self.arrival_rate_cell[kk]) * 1e6 * TS_duration
        traffic_observation = traffic_observation.reshape(n_cell * t_ttl)
        position_observation = np.array(S0_proj)  # # initial projection point of satellite on earth
        observation = np.concatenate((position_observation, traffic_observation))  # LEO projection position+traffic
        self.states.append(observation)
        self.render_states.append(self.states[-1])

        return observation, 0  # return current state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.reset()
        return [seed]

