import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class G:
    np.random.seed(0)  # Seed for random processes

    tau = 0.5  # Relaxation constant
    SOCIAL_DISTANCE = 2  # Social Distancing measure
    SOCIAL_DISTANCE_FORCE = 3  # Force between agents within social distance
    DT_FD = 0.001  # delta for finite difference calculation of gradient

    phi = 200.  # Field of view angle
    c = 0.5  # Field of View Constant
    V_ab0 = 2.3  # Agent avoidance magnitude constant
    SIGMA = 0.3  # Agent avoidance decay constant
    U_aB0 = 10.0  # Wall avoidance magnitude constant
    R = 0.2  # Wall avoidance decay constant

    ENVIRONMENT_SIZE = (50, 20)  # Size of corridor
    SPAWN_BUFFER = 2  # Buffer space between spawning agents at either side of corridor to edge

    DT = .1  # Simulation time-step (visualisation shows frames) in seconds
    SIMULATION_DURATION = 3600  # Duration of simulation in seconds
    current_simulation_time = 0.  # Current time (updated with DT)
    ANIMATION_PLAY_SPEED = 60  # Animation speed for visualisation
    COLOUR_PALETTE = LinearSegmentedColormap.from_list('rg', ["r", "g"], N=2)  # Red/green agent colours
    FPS = 20  # Video frame rate

    # Poisson distribution of arrivals (lambda = ave. arrivals / second). Note this is linearly increased
    poisson_lambda_range = (0.02, 0.2)
    poisson_lambda = poisson_lambda_range[0]
    next_arrival_time = current_simulation_time - np.log(1.0 - np.random.random()) / poisson_lambda
