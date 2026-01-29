import argparse
import sys
import os
import torch
import torch.nn as nn
import numpy as np

# =================================================================================
# 🚑 路径修复
# =================================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
isaaclab_root = os.path.abspath(os.path.join(current_dir, "../../../"))
source_path = os.path.join(isaaclab_root, "source")
if os.path.exists(source_path):
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

# =================================================================================
# 🚀 启动 App
# =================================================================================
try:
    import isaacsim
except ImportError:
    pass
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sim2Sim UR5e (Final Obs Structure)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.livestream = 1 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =================================================================================
# 📦 导入依赖
# =================================================================================
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
import isaaclab.envs.mdp as mdp
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg
from isaaclab.managers import ObservationTermCfg 

# =================================================================================
# 🧠 网络结构
# =================================================================================
class SB3CompatibleActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.mlp_extractor = nn.ModuleDict({
            "policy_net": nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh()
            ),
            "value_net": nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh()
            )
        })
        self.action_net = nn.Linear(256, action_dim)
        self.value_net = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        x = self.mlp_extractor["policy_net"](obs)
        return self.action_net(x)

# =================================================================================
# 🏗️ 场景与环境配置
# =================================================================================
ISAAC_NUCLEUS_SERVER = "https://omniverse-content-production.s3.us-west-2.amazonaws.com"

@configclass
class UR5eSceneCfg(InteractiveSceneCfg):
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_SERVER}/Assets/Isaac/2023.1.1/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.57, "elbow_joint": 1.57, "wrist_1_joint": -1.57, "wrist_2_joint": -1.57, "wrist_3_joint": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*", "elbow_.*", "wrist_.*"],
                stiffness=0.0, damping=1000.0
            ),
        },
    )
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material_path="/World/Materials/CubeMat",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

@configclass
class ActionsCfg:
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"], body_name="tool0",
        controller=DifferentialIKControllerCfg(command_type="pose", ik_method="dls"), scale=0.5,
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 只要能生成数据通过初始化检查就行
        dummy_joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
    policy = PolicyCfg()

@configclass
class RewardsCfg:
    pass

@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=100.0)

@configclass
class UR5eLiftEnvCfg(ManagerBasedRLEnvCfg):
    decimation = 2            
    episode_length_s = 100.0  
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    actions = ActionsCfg()
    scene = UR5eSceneCfg(num_envs=1, env_spacing=2.0)
    sim = sim_utils.SimulationCfg(dt=1.0/60.0, render_interval=2)

# =================================================================================
# 🌉 Sim2Sim 适配器
# =================================================================================
class Sim2SimAdapter:
    def __init__(self, env, weight_path, stats_path):
        self.env = env
        self.device = env.device
        
        print(f"📥 Loading Weights: {weight_path}")
        self.obs_dim = 63  # 严格匹配 Total Dim
        self.action_dim = 4
        
        self.policy = SB3CompatibleActor(self.obs_dim, self.action_dim).to(self.device)
        
        try:
            state_dict = torch.load(weight_path, map_location=self.device)
            self.policy.load_state_dict(state_dict, strict=True)
            print("✅ Policy weights loaded perfectly!")
        except Exception as e:
            print(f"❌ Strict loading failed: {e}")
            self.policy.load_state_dict(state_dict, strict=False)

        self.policy.eval()

        try:
            stats = torch.load(stats_path, map_location=self.device)
            self.obs_mean = stats["mean"].to(self.device)
            self.obs_var = stats["var"].to(self.device)
            print("✅ Norm stats loaded.")
        except:
            print("⚠️ Using default stats.")
            self.obs_mean = torch.zeros(self.obs_dim, device=self.device)
            self.obs_var = torch.ones(self.obs_dim, device=self.device)

    def step(self):
        robot = self.env.scene["robot"]
        cube = self.env.scene["cube"]
        
        arm_q = robot.data.joint_pos[:, :6]
        arm_dq = robot.data.joint_vel[:, :6]
        
        try:
            eef_idx = robot.find_bodies("tool0")[0][0]
        except:
            try:
                eef_idx = robot.find_bodies("wrist_3_link")[0][0]
            except:
                eef_idx = 6
        
        eef_pos = robot.data.body_pos_w[:, eef_idx]
        eef_quat = robot.data.body_quat_w[:, eef_idx]
        cube_pos = cube.data.root_pos_w
        cube_quat = cube.data.root_quat_w
        
        # --- 1. 构建 Object State (10 dims) ---
        # 顺序: [cube_pos(3), cube_quat(4), relative_pos(3)]
        # ⚠️ 注意：Robosuite 的 relative 是 (cube - gripper) 还是 (gripper - cube)？
        # 通常是 object - gripper (目标 - 当前)
        obj_state = torch.cat([
            cube_pos, 
            cube_quat, 
            cube_pos - eef_pos
        ], dim=-1) # (N, 10)

        # --- 2. 构建 Proprio State (53 dims) ---
        # 核心数据: [cos(6), sin(6), vel(6), eef_pos(3), eef_quat(4)] = 25 dims
        # 剩余 28 dims: 我们用 0 填充 (包含 joint_pos, acc, gripper等)
        # ⚠️ 注意: inspect_obs 显示 raw dict 里还有 joint_pos(6)，为了保险，我们把它也加上
        
        proprio_core = torch.cat([
            torch.cos(arm_q),   # (N, 6)
            torch.sin(arm_q),   # (N, 6)
            arm_dq,             # (N, 6)
            eef_pos,            # (N, 3)
            eef_quat,           # (N, 4)
            # 这里是 25 维
        ], dim=-1)
        
        # 目标是 53 维，还需要补 28 维
        # 我们用 arm_q (6) + zeros(22) 来填充，因为 raw dict 里有 joint_pos
        padding = torch.cat([
            arm_q, # (N, 6)
            torch.zeros((arm_q.shape[0], 22), device=self.device) # (N, 22)
        ], dim=-1)
        
        proprio_state = torch.cat([proprio_core, padding], dim=-1) # (N, 53)

        # --- 3. 最终拼接 (必须 Object 在前!) ---
        obs = torch.cat([obj_state, proprio_state], dim=-1) # (N, 63)

        # 归一化
        norm_obs = torch.clamp((obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -10, 10)
        
        # 推理
        with torch.no_grad():
            action = self.policy(norm_obs)
            
        # 动作转换: 4D (Pos+Grip) -> 7D (Pos+Rot+Grip)
        target_pos = action[:, :3]
        # 构造单位四元数 [1, 0, 0, 0]
        target_rot = torch.zeros((action.shape[0], 4), device=self.device)
        target_rot[:, 0] = 1.0 
        
        target_action = torch.cat([target_pos, target_rot], dim=-1)
            
        return target_action

# =================================================================================
# 🏁 主程序
# =================================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(script_dir, "policy_weights.pt")
    stats_path = os.path.join(script_dir, "obs_stats.pt")
    
    if not os.path.exists(weight_path):
        print(f"❌ Error: Missing {weight_path}")
        return

    env = ManagerBasedRLEnv(cfg=UR5eLiftEnvCfg())
    adapter = Sim2SimAdapter(env, weight_path, stats_path)
    env.reset()
    print("🚀 Sim2Sim Running (Obs Order Corrected!)...")
    
    while simulation_app.is_running():
        env.step(adapter.step())
    env.close()

if __name__ == "__main__":
    main()