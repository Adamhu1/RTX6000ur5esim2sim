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

parser = argparse.ArgumentParser(description="Sim2Sim UR5e (Action Dim Fix)")
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
    # command_type="pose" 需要 7 维输入 (3 Pos + 4 Quat)
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"], body_name="tool0",
        controller=DifferentialIKControllerCfg(command_type="pose", ik_method="dls"), scale=0.5,
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
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
        self.obs_dim = 63
        self.action_dim = 4
        
        self.policy = SB3CompatibleActor(self.obs_dim, self.action_dim).to(self.device)
        
        try:
            state_dict = torch.load(weight_path, map_location=self.device)
            self.policy.load_state_dict(state_dict, strict=True)
            print("✅ Policy weights loaded perfectly!")
        except Exception as e:
            print(f"❌ Strict loading failed, trying non-strict: {e}")
            self.policy.load_state_dict(state_dict, strict=False)

        self.policy.eval()

        try:
            stats = torch.load(stats_path, map_location=self.device)
            mean_from_file = stats["mean"].to(self.device)
            var_from_file = stats["var"].to(self.device)
            
            if mean_from_file.shape[0] != self.obs_dim:
                print(f"⚠️ Norm stats dim mismatch. Padding with zeros.")
                self.obs_mean = torch.zeros(self.obs_dim, device=self.device)
                self.obs_var = torch.ones(self.obs_dim, device=self.device)
                min_dim = min(mean_from_file.shape[0], self.obs_dim)
                self.obs_mean[:min_dim] = mean_from_file[:min_dim]
                self.obs_var[:min_dim] = var_from_file[:min_dim]
            else:
                self.obs_mean = mean_from_file
                self.obs_var = var_from_file
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
        
        gripper_fake = torch.zeros((arm_q.shape[0], 6), device=self.device)
        
        base_obs = torch.cat([
            torch.cos(arm_q), torch.sin(arm_q), 
            arm_dq, 
            eef_pos, 
            torch.cat([eef_quat[:, 1:], eef_quat[:, 0:1]], dim=-1), 
            gripper_fake, gripper_fake, 
            cube_pos, 
            torch.cat([cube_quat[:, 1:], cube_quat[:, 0:1]], dim=-1), 
            cube_pos - eef_pos 
        ], dim=-1)

        padding_size = self.obs_dim - base_obs.shape[-1]
        if padding_size > 0:
            padding = torch.zeros((base_obs.shape[0], padding_size), device=self.device)
            obs = torch.cat([base_obs, padding], dim=-1)
        else:
            obs = base_obs

        norm_obs = torch.clamp((obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -10, 10)
        
        with torch.no_grad():
            action = self.policy(norm_obs)
            
        # ⚠️ 关键修正：DiffIK Pose 模式需要 7 维动作 (Pos + Rot)
        # 模型输出 action shape: (N, 4) -> [x, y, z, grip?]
        # 我们只取前3维位置
        target_pos = action[:, :3]
        
        # 构造 4D 单位四元数 [1, 0, 0, 0] (w, x, y, z) 代表不旋转
        # shape: (N, 4)
        target_rot = torch.zeros((action.shape[0], 4), device=self.device)
        target_rot[:, 0] = 1.0 # 设置 w = 1.0
        
        # 3 (Pos) + 4 (Quat) = 7
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
    print("🚀 Sim2Sim Running (Dimensions 7D Corrected!)...")
    
    while simulation_app.is_running():
        env.step(adapter.step())
    env.close()

if __name__ == "__main__":
    main()