# remote_browse.py
import argparse
from omni.isaac.lab.app import AppLauncher

# 1. 配置启动参数
parser = argparse.ArgumentParser(description="Remote Asset Browser")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 强制开启相机和直播功能
args_cli.enable_cameras = True
args_cli.headless = True # 服务器端不显示窗口

# 2. 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. 导入必要模块 (必须在启动后)
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from omni.isaac.lab.utils import configclass

# 定义一个什么都没有的空环境
@configclass
class EmptyEnvCfg(ManagerBasedRLEnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    sim = SimulationCfg(dt=0.01)

def main():
    # 初始化环境
    env = ManagerBasedRLEnv(cfg=EmptyEnvCfg())
    
    print("--------------------------------------------")
    print("🚀 远程 GUI 已启动！")
    print("👉 请在本地打开 Isaac Sim WebRTC Client")
    print("👉 输入服务器 IP 进行连接")
    print("👉 连接成功后，在菜单栏 Window -> Browsers -> Content 查找 UR5e")
    print("--------------------------------------------")

    # 死循环，保持画面一直开着供你操作
    while simulation_app.is_running():
        env.step(action=None) # 空转

    env.close()

if __name__ == "__main__":
    main()