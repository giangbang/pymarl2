from functools import partial
import sys
import os
from .smaclite_wrapper import SMACliteWrapper

from .multiagentenv import MultiAgentEnv

# from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except Exception as e:
    gfootball = False
    print(e)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

def smaclite_fn(**kwargs) -> MultiAgentEnv:

    return SMACliteWrapper(**kwargs)

def gymma_fn(**kwargs) -> MultiAgentEnv:
    from .gymma import GymmaWrapper
    return GymmaWrapper(**kwargs)

REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["gymma"] = gymma_fn
REGISTRY["smaclite"] = smaclite_fn


if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
