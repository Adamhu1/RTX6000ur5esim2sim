# scripts/test_sim.py
import sys
import os

# ⚠️⚠️⚠️ 这里的路径必须替换为你第一步 find 命令找到的真实路径！！！
# 如果你不知道，请先运行 find 命令
isaac_sim_path = "/home/weihao/.local/share/ov/pkg/isaac_sim-2023.1.1" 

if os.path.exists(isaac_sim_path):
    sys.path.append(isaac_sim_path)
    print(f"✅ 强制添加路径: {isaac_sim_path}")
else:
    print(f"❌ 警告: 路径不存在: {isaac_sim_path}")

# 尝试导入
try:
    from omni.isaac.lab.app import AppLauncher
    print("✅ 成功导入 AppLauncher!")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("当前 sys.path:")
    for p in sys.path: print(p)
    sys.exit(1)

import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.livestream = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("🚀 仿真器启动成功！")
while simulation_app.is_running():
    simulation_app.update()