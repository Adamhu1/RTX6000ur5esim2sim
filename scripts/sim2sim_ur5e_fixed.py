# scripts/sim2sim_ur5e_fixed.py
import argparse
import sys
import os

# --- 1. 最先导入 AppLauncher (必须在 torch/numpy 之前!) ---
from omni.isaac.lab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="Sim2Sim: Robosuite UR5e to Isaac Lab")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 强制开启直播 (既然你验证了这个参数有效)
args_cli.livestream = 1 

# --- 2. 启动仿真 App ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 3. App 启动后，才能导入其他库 ---
import torch
import numpy as np

# Isaac Lab 模块
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.controllers.differential_ik import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass

# SB3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# =============================================================================
# 配置与类定义 (保持不变，记得修改 usd_path!)
# =============================================================================

@configclass
class UR5eLiftEnvCfg(ManagerBasedRLEnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    
    # 机器人
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ArticulationCfg.utils.SpawnUsdFileCfg(
            # ⚠️ 这里填你刚才确认能找到的路径
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd", 
            activate_joint_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
            },
        ),
        actuators={
            "arm": ArticulationCfg.ActuatorsCfg(
                joint_names_expr=["shoulder_.*", "elbow_.*", "wrist_.*"],
                effort_limit=300.0, velocity_limit=10.0, stiffness=0.0, damping=1000.0,
            ),
            "gripper": ArticulationCfg.ActuatorsCfg(
                joint_names_expr=["finger_joint"],
                effort_limit=100.0, velocity_limit=2.0, stiffness=1000.0, damping=10.0,
            ),
        },
    )

    # 物体
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=RigidObjectCfg.utils.SpawnCuboidCfg(
            size=(0.04, 0.04, 0.04),
            mass=0.1,
            visual_material_path="/World/Materials/CubeMat",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
    )

    # 动作 (IK)
    actions = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"],
        body_name="ee_link", # ⚠️ 如果报错找不到 body，请检查 USD 文件里的末端是不是叫 tool0
        controller_cfg=DifferentialIKControllerCfg(command_type="pose", command_curr_pose=True),
        scale=0.5,
    )
    
    sim = SimulationCfg(dt=1.0/60.0, render_interval=2)

class Sim2SimAdapter:
    def __init__(self, env, model_path, pkl_path):
        self.env = env
        self.device = env.device
        # 这里的路径需要注意：如果是相对路径，要确保文件在运行目录下
        # 建议写绝对路径，防止找不到文件
        print(f"📥 Loading Model from: {model_path}")
        self.model = PPO.load(model_path, device="cpu")
        
        dummy_env = lambda: None; dummy_env.observation_space = None
        self.vec_norm = VecNormalize.load(pkl_path, dummy_env)
        self.obs_mean = torch.tensor(self.vec_norm.obs_rms.mean, device=self.device).float()
        self.obs_var = torch.tensor(self.vec_norm.obs_rms.var, device=self.device).float()
        self.epsilon = 1e-8

    def get_robosuite_obs(self):
        # ... (保持原有的观测拼接逻辑不变) ...
        # 为节省篇幅，这里复用之前的 get_robosuite_obs 逻辑
        # 核心逻辑：获取 Isaac 数据 -> 拼成 47维 向量
        robot = self.env.scene["robot"]
        cube = self.env.scene["cube"]
        
        # [简写] 数据获取
        arm_q = robot.data.joint_pos[:, :6]
        arm_dq = robot.data.joint_vel[:, :6]
        gripper_q = robot.data.joint_pos[:, 6:7]
        gripper_dq = robot.data.joint_vel[:, 6:7]
        
        eef_idx = robot.find_bodies("ee_link")[0]
        eef_pos = robot.data.body_pos_w[:, eef_idx]
        eef_quat = robot.data.body_quat_w[:, eef_idx] # w,x,y,z
        
        cube_pos = cube.data.root_pos_w
        cube_quat = cube.data.root_quat_w
        
        # 拼接
        obs_vec = torch.cat([
            torch.cos(arm_q), torch.sin(arm_q), arm_dq,
            eef_pos, 
            torch.cat([eef_quat[:, 1:], eef_quat[:, 0:1]], dim=-1), # xyzw
            gripper_q.repeat(1, 6), gripper_dq.repeat(1, 6),
            cube_pos, 
            torch.cat([cube_quat[:, 1:], cube_quat[:, 0:1]], dim=-1), # xyzw
            cube_pos - eef_pos
        ], dim=-1)
        return obs_vec

    def step(self):
        obs = self.get_robosuite_obs()
        norm_obs = torch.clamp((obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon), -10, 10)
        with torch.no_grad():
            action_np, _ = self.model.policy.predict(norm_obs.cpu().numpy(), deterministic=True)
        
        action_tensor = torch.tensor(action_np, device=self.device)
        # 这里的 Action 需要根据你的 Config 调整
        return torch.cat([action_tensor[:, :3], torch.zeros(action_tensor.shape[0], 3, device=self.device)], dim=-1)

def main():
    env_cfg = UR5eLiftEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # ⚠️ 确保这两个文件也在 scripts 目录下，或者写绝对路径
    # 假设你把模型文件也放到了 scripts/ 目录下
    adapter = Sim2SimAdapter(env, "ur_lift_fixed_final.zip", "ur_lift_fixed_vecnormalize.pkl")

    env.reset()
    print("🚀 Sim2Sim Running...")
    
    while simulation_app.is_running():
        ik_action = adapter.step()
        env.step(ik_action)

    env.close()

if __name__ == "__main__":
    main()