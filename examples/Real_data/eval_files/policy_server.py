"""
 Wirtten by: 范斌
 Date: 2025/7/24
 Description:
    该代码负责通过网络，接受机器人端传输来的观测数据，并调用模型进行动作预测，然后将动作发送给机器人端。
    首先所有模型统一放在 policy 文件夹中，并在里面写入一个 python 文件，实现模型的封装好的类，同时该 python 文件的名称应该和里面实现的类的名称一致，例如：
    
    在 policy/3D-Diffusion-Policy/DP3.py 里面的类的名字就叫 DP3.
    在 policy/RDT/RDT.py 里面的类的名字就叫 RDT.

    然后你应该为这个类实现 4 个基本函数：
        1. __init__(self, usr_args), 用于初始化模型，接收用户输入的参数；
        2. process_and_update_obs(self, obs), 用于处理输入数据并更新观测；
        3. predict_actions(self), 用于预测动作;
        4. reset(self), 用于重置模型的观测。
    
    最后在模型文件夹中，实现 run.sh 文件，用于传入模型必要的参数并启动 server.
    
Host isee_target
  HostName 192.168.1.101
  User robotic
  IdentityFile ~/.ssh/robot_rsa
  ProxyJump isee_jump_80
  LocalForward 65432 192.168.1.101:65432
  RemoteForward 65433 localhost:65433

"""
import socket
import pickle
from termcolor import cprint
import importlib
import numpy as np

import logging
import argparse
from examples.Real_data.eval_files.model_interface import ModelClient
import torch, os
from PIL import Image


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--use_bf16", action="store_true")
    return parser


class Policy_server:
    """
    保证 server_port与 robot_port 与配置文件里一致
    """
    def __init__(self, args, server_port=65468, robot_port=65435):
        
        self.policy = ModelClient(args.ckpt_path, args.output_dir)

        self.port = server_port
        self.robot_port = robot_port
        self.running = True # 控制服务器运行的标志

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen()
            s.settimeout(10000)  # 设置 accept 超时，以便定期检查 running 状态
            cprint(f"Server is listening on port {self.port}...", 'green')

            while self.running:
                try:
                    conn, addr = s.accept()

                    with conn:
                        data = b''
                        while True:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            data += chunk
                        # 判断是否是关闭指令
                        if data == b'close':
                            self.running = False
                            cprint("Received close command from client.", 'yellow')
                            # 可以在这里执行清理操作
                            break

                        try:
                            data_dict = pickle.loads(data)
                            command = data_dict['command']
                            if command == 'inference':
                                actions = self.get_actions(data_dict)
                                self.send_actions(actions)
                            elif command == 'reset':
                                cprint('observation reset...', 'cyan')
                                self.policy.reset()
                            elif command == 'update':
                                # 更新观测
                                self.policy.process_and_update_obs(data_dict)
                            else:
                                raise ValueError("Unknown command: {}".format(command))
                            
                            
                        except Exception as e:
                            print("Failed to load .npz data:", e)

                except socket.timeout:
                    cprint(f'Time out!!!', 'yellow')
                    break  # 超时退出
                except Exception as e:
                    print("Error during connection handling:", e)

            cprint("Server is shutting down.", 'red')

    def get_actions(self, data_dict):
        actions = self.policy.predict_actions(data_dict)
        return actions

    def send_actions(self, actions):
        actions_data = pickle.dumps(actions)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('localhost', self.robot_port))  # 客户端监听的端口
                cprint("Sending actions to client...", 'green')
                s.sendall(actions_data)
                cprint("Actions sent.", 'green')
        except ConnectionRefusedError:
            cprint("Error: Client is not listening for actions. Is the client ready?", 'yellow')
        except Exception as e:
            cprint(f"Error sending actions: {e}", 'red')
    
    def make_ramdon_input(self):
        # 生成测试数据
        obs = {}
        obs['main_camera_rgb'] = (np.random.rand(720, 1280, 3)*255).astype(np.uint8)  # (720, 1280, 3)
        obs['wrist_camera_rgb'] = (np.random.rand(720, 1280, 3)*255).astype(np.uint8)
        obs['robot_joints_angle'] = np.ones((8), dtype=np.float32)       # (8): 关节角 + 夹爪状态
        obs['robot_endpose'] = np.ones((8), dtype=np.float32)  # (8): xyz + xyzw(四元数) + 夹爪状态
        return obs

    def test(self):
        # 测试数据格式是否对齐
        obs = self.make_ramdon_input()
        data_dict = self.get_actions(obs)
        print(data_dict['actions'])
        cprint('Test over', 'green')
        self.policy.reset()
        return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    server = Policy_server(args)
    server.test()
    server.run()
