from gym.envs.registration import register

register(
    id='BeamHopping-v0',
    entry_point='gym_beamhopping.envs:BeamHoppingEnv',
)