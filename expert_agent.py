import os
import time
import logging
import grid2op
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Observation import BaseObservation
from grid2op.Runner import Runner

from lightsim2grid.lightSimBackend import LightSimBackend

from l2rpn_baselines.PPO_SB3.utils import SB3Agent

from package.modules.topology_heuristic import RecoPowerlineModule, RecoverInitTopoModule
from package.modules.convex_optim import OptimModule
from package.modules.rewards import MaxRhoReward
from package.modules.topology_nn_policy import TopoNNTopKModule

from expert_utils import make_gymenv, create_env_op

logging.basicConfig(level=logging.INFO, filename="agents.log", filemode="w")
logger = logging.getLogger(__name__)


L2RPN_CASE14_DEFAULT_OPTIM_CONFIG = {
    "margin_th_limit": 0.93,
    "alpha_por_error": 0.5,
    "rho_danger": 0.99,
    "rho_safe": 0.9,
    "penalty_curtailment_unsafe": 15,
    "penalty_redispatching_unsafe": 0.005,
    "penalty_storage_unsafe": 0.0,
    "penalty_curtailment_safe": 0.0,
    "penalty_redispatching_safe": 0.0,
    "penalty_storage_safe": 0.0,
    "weight_redisp_target": 1.0,
    "weight_storage_target": 1.0,
    "weight_curtail_target": 1.0,
    "margin_rounding": 0.01,
    "margin_sparse": 5e-3,
    "max_iter": 100000,
    "areas": False,
    "sim_range_time_step": 1,
}

class ExpertAgent(BaseAgent):
    def __init__(
        self,
        action_space: ActionSpace,
        env,
        env_gym,
        model_path: str,
        top_k: int = 10,
        rho_danger: float = 0.99,
        rho_safe: float = 0.9,
    ):
        BaseAgent.__init__(self, action_space=action_space)
        # Environment
        self.env = env
        self.env_gym = env_gym
        self.rho_danger = rho_danger
        self.rho_safe = rho_safe
        # Sub-modules
        # Heuristic
        self.reconnect = RecoPowerlineModule(self.action_space)
        self.recover_topo = RecoverInitTopoModule(self.action_space)
        # Continuous control
        self.optim = OptimModule(env, self.action_space, config=L2RPN_CASE14_DEFAULT_OPTIM_CONFIG)
        
        # TopoNN
        self.topo_agent = TopoNNTopKModule(
            self.action_space,
            env_gym,
            model_path=model_path,
            top_k=top_k
        )
        # self.topo_agent = SB3Agent(env.action_space,
        #                            env_gym.action_space,
        #                            env_gym.observation_space,
        #                            nn_path=model_path,
        #                            nn_kwargs=None)
        self.action_list = []

    def act(
        self, observation: BaseObservation, reward: float, done: bool = False
    ) -> BaseAction:
        start = time.time()

        # Init action with "do nothing"
        act = self.action_space({})

        # Try to perform reconnection if necessary
        reconnect_act = self.reconnect.get_act(observation, act, reward)
        _obs, _rew, _done, _info = observation.simulate(reconnect_act, time_step=1)
        # logger.info(f"Info: {_info}")
        # logger.info(f"Exception: {}")
        
        if reconnect_act is not None:  
            if (reconnect_act is not None
                and not _done
                and reconnect_act != self.action_space({})
                and 0. < _obs.rho.max() < 2.
                and (len(_info["exception"]) == 0)
            ):
                logger.info("calling reconnection module")
                act += reconnect_act

        if observation.rho.max() > self.rho_danger:
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=self.rho_danger
            )
            if recovery_act is not None:
                logger.info("Calling recovery action")
                act += recovery_act
            else:
                if all(observation.line_status):
                    # TODO : this if condition could be useful for implementation of domains shift KPI
                    # gym_obs = self.env_gym.observation_space.to_gym(observation)
                    # topo_act = self.topo_agent.get_act(gym_obs, act, reward)
                    # topo_act = self.env_gym.action_space.from_gym(topo_act)
                    topo_act = self.topo_agent.get_act(observation, act, reward)
                    
                    if topo_act is not None:
                        logger.info("Calling topo agent")
                        _obs, *_ = observation.simulate(act + topo_act)
                        logger.info(topo_act)
                        act += topo_act
                    
            _obs, _rew, _done, _info = observation.simulate(act, time_step=1)
            if _obs.rho.max() > self.rho_safe or (len(_info["exception"]) != 0):
                logger.info("calling optim module")
                # logger.warning(f"EXCEPTION : {_info['exception']}")
                # logger.warning(f"Done: {_done}")
                # act = self.action_space({})
                act = self.optim.get_act(observation, act, reward)
            
        elif _obs.rho.max() < self.rho_safe:
            # Try to find a recovery action when the grid is safe
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=0.8
            )
            if recovery_act is not None:
                logger.info("Calling recovery action (grid is safe)")
                act += recovery_act
        if act != self.action_space({}):
            self.action_list.append(act)
        return act

def run_agent(env, env_gym, model_path, top_k=20):
    agent = ExpertAgent(env.action_space, 
                        env, 
                        env_gym, 
                        model_path=model_path,
                        top_k=top_k)
    
    verbose = True
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose
    logs_path = "logs"

    # Build the runner
    runner = Runner(**runner_params, 
                    agentClass=None, 
                    agentInstance=agent)

    if logs_path is not None:
        os.makedirs(logs_path, exist_ok=True)
        
    results = runner.run(
        path_save=logs_path,
        nb_episode=2,
        nb_process=1,
        max_iter=env.chronics_handler.max_episode_duration(),
        pbar=verbose,
        env_seeds=[0, 1]
    )
    
    return agent, results

if __name__ == "__main__":
    env_name = 'l2rpn_case14_sandbox'
    reward_class = MaxRhoReward
    seed = 122435
    env = grid2op.make(env_name,
                       backend = LightSimBackend(), 
                       reward_class = reward_class)
    
    env.seed(seed)
    obs = env.reset()
    
    env_gym = make_gymenv(env)
    
    env_op, env_gym_op = create_env_op(env_name, reward_class, seed)
    
    model_path = "model_logs"
    model_name = "PPO_SB3"
    final_model_path = os.path.join(model_path, model_name, f"{model_name}.zip")
    
    # Evaluate on normal distribution
    agent, results = run_agent(env, env_gym, final_model_path, top_k=20)
    
    # Evaluate on Data drift
    agent_op, results_op = run_agent(env_op, env_gym_op, final_model_path, top_k=20)
        
    # Evaluate the Fine tuned model on data drift
    final_model_path = os.path.join(model_path, model_name, f"{model_name}_FINETUNED.zip")
    agent_f, results_f = run_agent(env, env_gym, final_model_path, top_k=20)
    agent_op_f, results_op_f = run_agent(env_op, env_gym_op, final_model_path, top_k=20)
    