import grid2op
from grid2op.Environment import Environment
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpaceGymnasium
# for oponnent line disconnection
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget
from lightsim2grid import LightSimBackend

from l2rpn_baselines.PPO_SB3.utils import remove_non_usable_attr

def make_gymenv(env: Environment, 
                obs_attr_to_keep=["rho"], 
                act_to_keep=("set_bus",)):
    """Create a gymnasium environment from grid2op

    Parameters
    ----------
    env : `Environment`
        A grid2op.env
    obs_attr_to_keep : list, optional
        the list of attributes to keep for an observation, by default ["rho"]
    act_to_keep : tuple, optional
        the list of action types to include in the gym environment action space, by default ("set_bus",)

    Returns
    -------
    _type_
        _description_
    """    
    act_attr_to_keep = remove_non_usable_attr(env, act_to_keep)
    # print("****************", act_attr_to_keep)
    env_gym = GymEnv(env)
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(env.observation_space,
                                               attr_to_keep=obs_attr_to_keep)
    env_gym.action_space.close()
    env_gym.action_space = DiscreteActSpaceGymnasium(env.action_space,
                                                     attr_to_keep=act_attr_to_keep)
    
    return env_gym

def create_env_op(env_name, reward_class, seed):
    """Create the opponent environment with line attacks
    
    It is used to evaluate the capability of the agent when encountering data drift

    Parameters
    ----------
    env_name : _type_
        _description_
    seed : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    kwargs_opponent={"lines_attacked": ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]}

    env = grid2op.make(env_name,
                       backend=LightSimBackend(),
                       # chronics_class=ChangeNothing
                       # chronics_class=ChangeNothing,
                       opponent_attack_cooldown=12*24, # 12 time stamps per hour (every 5 minutes), 1 attack per day is authorized
                       opponent_attack_duration=12*4, # 4 hours the delay to be able to reconnect the line
                       opponent_action_class=PowerlineSetAction,
                       opponent_class=RandomLineOpponent,
                       opponent_budget_class=BaseActionBudget,
                       opponent_budget_per_ts=0.5, # The higher this number, the faster the the opponent will regenerate its budget.
                       opponent_init_budget=0,  # It is set to 0 to “give” the agent a bit of time before the opponent is triggered.
                       kwargs_opponent=kwargs_opponent,
                       reward_class=reward_class
                      )
    env.seed(seed=seed)
    
    env_gym = make_gymenv(env)
    
    return env, env_gym
