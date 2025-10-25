import os
import re
import grid2op
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv, MultiDiscreteActSpaceGymnasium, DiscreteActSpaceGymnasium
from lightsim2grid import LightSimBackend

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from l2rpn_baselines.PPO_SB3.utils import SB3Agent

from l2rpn_baselines.PPO_SB3.utils import (default_obs_attr_to_keep, 
                                           default_act_attr_to_keep,
                                           remove_non_usable_attr,
                                           save_used_attribute)

from package.modules.rewards import PPO_Reward


if __name__ == "__main__":
    env = grid2op.make("l2rpn_case14_sandbox", 
                       backend=LightSimBackend(), 
                       reward_class=PPO_Reward)
    
    env.chronics_handler.real_data.set_filter(lambda x: re.match(".*0$", x) is not None)
    env.chronics_handler.real_data.reset()
    
    obs_attr_to_keep = ["rho"]
    # ['set_line_status', 'change_line_status', 'set_bus', 'change_bus', 'raise_alarm', 'raise_alert', 
    # 'sub_set_bus', 'sub_change_bus', 'one_sub_set', 'one_sub_change', 'one_line_set', 'one_line_change']
    # - change_bus
    # - change_line_status
    # - curtail
    # - curtail_mw
    # - redispatch
    # - set_bus
    # - set_line_status
    # - set_line_status_simple
    # - set_storage
    act_attr_to_keep = remove_non_usable_attr(env, ["set_bus"])
    
    
    
    env_gym = GymEnv(env)
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(env.observation_space,
                                               attr_to_keep=obs_attr_to_keep)
    env_gym.action_space.close()
    env_gym.action_space = DiscreteActSpaceGymnasium(env.action_space,
                                          attr_to_keep=act_attr_to_keep)
    
    name = "PPO_SB3"
    net_arch=[200, 200, 200]
    policy_kwargs = {}
    policy_kwargs["net_arch"] = net_arch
    # save_every_xxx_steps=2000
    # eval_every_xxx_steps=1000
    # kwargs = {}
    # kwargs["save_every_xxx_steps"] = save_every_xxx_steps
    logs_dir = "model_logs"
    if logs_dir is not None:
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        this_logs_dir = os.path.join(logs_dir, name)
        
    nn_kwargs = {
            "policy": MlpPolicy,
            "env": env_gym,
            "verbose": True,
            "learning_rate": 3e-4,
            "tensorboard_log": this_logs_dir,
            "policy_kwargs": policy_kwargs,
            "device": "auto"
    }
        
    # nn_kwargs = None
    agent = SB3Agent(env.action_space,
                     env_gym.action_space,
                     env_gym.observation_space,
                    #  nn_path=os.path.join(this_logs_dir, f"{name}.zip"),
                     nn_kwargs=nn_kwargs)
    
    # train it
    # agent.nn_model.learn(total_timesteps=1000, progress_bar=True)
    
    # save the model
    # agent.nn_model.save(os.path.join(this_logs_dir, name))
    
    # Load the model
    # agent.load(os.path.join(this_logs_dir, f"{name}.zip"))
    
    # # predict an action
    # # action = env_gym.action_space.sample()
    # action = agent.get_act(env_gym.observation_space.sample(), None, None)
    # action = env_gym.action_space.from_gym(action)
    # print(action)
    
    # env_gym.close()
    