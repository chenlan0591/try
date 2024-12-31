"""
This file contains all the constant configurations
"""
import numpy as np
import math

n_cell = 19  # number of cells
n_beam = 4  # number of beams, N_beam<N_cell
p_total = 1e3  # total power=30 dBW = 10^3 W
b_total = 500 * 1e6  # total bandwidth=500 MHz
max_lambda = 150  # maximum arrival rate=150 Mbps
min_lambda = 50  # minimum arrival rate=50 Mbps
max_gain_tx = 38.5  # dBi, maximum transmit antenna gain at satellite
gain_rx = 31.6  # dBi, received antenna gain at terminals
noise_power = 6.92 * 1e-21  # N_0, -171.6 dBm/Hz
t_ttl = 40  # T_ttl=40 TSs
TS_per_episode = 500
TS_duration = 0.02  # duration of each TS = 20 ms
weight_factor = 0.5  # weight factor between throughput and delay fairness
max_throughput = 3e5
max_delay_fair = 20
r_E = 6370 * 1e3  # radius of the earth
d_S = 600 * 1e3  # satellite orbit height
v_S = 7.6 * 1e3  # velocity of satellite
location_factor = np.random.uniform(0, 1, n_cell)  # location weight factors for cells
r_C = 10 * 1e3  # radius of cells = 10 km
sqrt3 = math.sqrt(3)
location_cells = [[-3 * r_C, sqrt3 * r_C], [-3 * r_C, 0],
                  [-3 * r_C, -sqrt3 * r_C], [-1.5 * r_C, 1.5 * sqrt3 * r_C],
                  [-1.5 * r_C, 0.5 * sqrt3 * r_C], [-1.5 * r_C, -0.5 * sqrt3 * r_C],
                  [-1.5 * r_C, -1.5 * sqrt3 * r_C], [0, 2 * sqrt3 * r_C],
                  [0, sqrt3 * r_C], [0, 0], [0, -sqrt3 * r_C], [0, -2 * sqrt3 * r_C],
                  [1.5 * r_C, 1.5 * sqrt3 * r_C], [1.5 * r_C, 0.5 * sqrt3 * r_C],
                  [1.5 * r_C, -0.5 * sqrt3 * r_C], [1.5 * r_C, -1.5 * sqrt3 * r_C],
                  [3 * r_C, sqrt3 * r_C], [3 * r_C, 0], [3 * r_C, -sqrt3 * r_C]]
S0_proj = [-4 * r_C, -7 * sqrt3 * r_C / 3]  # initial projection point of satellite on earth
distance_TS = r_E * TS_duration * v_S / (d_S + r_E)  # motive distance of satellite projection point in each TS
f_c = 20 * 1e9  # carrier frequency=20 GHz
c = 3 * 1e8  # light speed
lambda_c = c / f_c  # wavelength
