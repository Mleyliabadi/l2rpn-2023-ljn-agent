import os
import re
import grid2op
from grid2op.gym_compat import GymEnv
from grid2op.Environment import Environment

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
from expert_utils import create_env_op, make_gymenv

def train_model(env: Environment,
                env_gym: GymEnv,
                name:str = "PPO_SB3", 
                save_path: str = "model_logs"):
    net_arch=[200, 200, 200]
    policy_kwargs = {}
    policy_kwargs["net_arch"] = net_arch
    # save_every_xxx_steps=2000
    # eval_every_xxx_steps=1000
    # kwargs = {}
    # kwargs["save_every_xxx_steps"] = save_every_xxx_steps
        
    nn_kwargs = {
            "policy": MlpPolicy,
            "env": env_gym,
            "verbose": True,
            "learning_rate": 3e-4,
            "tensorboard_log": save_path,
            "policy_kwargs": policy_kwargs,
            "device": "auto"
    }
        
    agent = SB3Agent(env.action_space,
                     env_gym.action_space,
                     env_gym.observation_space,
                     nn_kwargs=nn_kwargs)
    
    # train it
    agent.nn_model.learn(total_timesteps=1000, progress_bar=True)
    
    # save the model
    agent.nn_model.save(os.path.join(save_path, name))
    
    return agent

def fine_tune_model(env: Environment, 
                    env_gym: GymEnv, 
                    name: str, 
                    load_path: str,
                    total_time_steps=1000):
    print("Finetuning of a trained model")
    agent = SB3Agent(env.action_space,
                     env_gym.action_space,
                     env_gym.observation_space,
                     nn_path=os.path.join(load_path, f"{name}.zip"),
                     nn_kwargs=None)
    
    # load the PPO model properly
    model = PPO.load(path=os.path.join(load_path, f"{name}.zip"),
                     custom_objects = {'observation_space' : env_gym.observation_space, 
                                       'action_space' : env_gym.action_space})
    model.set_env(env_gym)
    
    # resume the training on new scenarios (Fine tune)
    model.learn(total_timesteps=total_time_steps, progress_bar=True)
    
    # save the fine tuned model
    model.save(os.path.join(load_path, name+"_FINETUNED"))
    
    agent.nn_model = model
    
    return agent

if __name__ == "__main__":
    env_name = "l2rpn_case14_sandbox"
    reward_class = PPO_Reward
    seed = 1234
    env = grid2op.make(env_name, 
                       backend=LightSimBackend(), 
                       reward_class=reward_class)
    env.seed(seed)
    obs_attr_to_keep = ["rho"]
    # - change_bus
    # - change_line_status
    # - curtail
    # - curtail_mw
    # - redispatch
    # - set_bus
    # - set_line_status
    # - set_line_status_simple
    # - set_storage
    act_attr_to_keep = ["set_bus"]
    env_gym = make_gymenv(env, obs_attr_to_keep, act_attr_to_keep)
    
    env.chronics_handler.real_data.set_filter(lambda x: re.match(".*0$", x) is not None)
    env.chronics_handler.real_data.reset()
    
    env_op, env_gym_op = create_env_op(env_name=env_name,
                                       reward_class=reward_class,
                                       seed=seed)
    
    logs_dir = "model_logs"
    if logs_dir is not None:
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        model_path = os.path.join(logs_dir, "PPO_SB3")
    
    
    # to indicate if the model is in finetuning step or training
    # TODO: using argparse library of python
    fine_tune = True
    
    if fine_tune == True:
        # Fine tune on the env with distribution shift (presence of opponent)
        agent = fine_tune_model(env_op, 
                                env_gym_op, 
                                name="PPO_SB3", 
                                load_path=model_path,
                                total_time_steps=100000)
    else:
        agent = train_model(env, env_gym, name="PPO_SB3_test", save_path=model_path)
    
    # Load the model
    # agent.load(os.path.join(this_logs_dir, f"{name}.zip"))
    
    # # predict an action
    # # action = env_gym.action_space.sample()
    # action = agent.get_act(env_gym.observation_space.sample(), None, None)
    # action = env_gym.action_space.from_gym(action)
    # print(action)
    
    # env_gym.close()
    