import numpy as np
from global_variables import G
from sim_backend import Agents, Animator


G.seed = 0
agents = Agents()

# If at steady steady state 1/ lambda = # people / length of corridor / v_pref (i.e, inflow is equal to outflow)
# then the number of people in the system is width of corridor * v_pref / lambda (we have v_pref = 1)
expected_count = G.ENVIRONMENT_SIZE[0] * G.poisson_lambda

# Loop to create multiple random agents
while agents.agents_count < expected_count:
    r0 = np.random.choice(np.linspace(0 + G.SPAWN_BUFFER, G.ENVIRONMENT_SIZE[0] - G.SPAWN_BUFFER))
    r1 = np.random.choice(np.linspace(0 + G.SPAWN_BUFFER, G.ENVIRONMENT_SIZE[1] - G.SPAWN_BUFFER))
    v0 = np.random.choice(np.linspace(-1, 1))
    v1 = np.random.choice(np.linspace(-1, 1))
    g = np.random.choice([0, 1])
    agents.add_agents(r=(r0, r1), v=(v0, v1), goal=g)

animator = Animator(agents, file_name="Example.mp4")
animator.run()
