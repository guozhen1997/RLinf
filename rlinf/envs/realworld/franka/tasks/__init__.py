from gymnasium.envs.registration import register
from rlinf.envs.realworld.franka.tasks.peg_insertion_env import PegInsertionEnv

register(
    id="PegInsertionEnv-v1", 
    entry_point="rlinf.envs.realworld.franka.tasks:PegInsertionEnv", 
)