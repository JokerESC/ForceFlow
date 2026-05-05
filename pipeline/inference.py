import os
import argparse
import json
import time
import traceback
from pathlib import Path

from pipeline.train import MultiViewResnetWithLowdimObsSeqCondition, load_config
import cv2
import numpy as np
import torch
from CleanDiffuser.cleandiffuser.diffusion import ContinuousRectifiedFlow
from CleanDiffuser.cleandiffuser.nn_diffusion import DiT1d
from CleanDiffuser.cleandiffuser.utils import MinMaxNormalizer
from termcolor import cprint
from torchvision.transforms.v2 import Normalize, Resize

from env.spacemouse import SpacemouseAgent
from env.xarm_env import XarmEnv
from env.realsense_env import RealsenseEnv
import atexit

from scripts.collect import KeyboardListener
import sys


CONTROL_FREQ = 30.0

RECORD_VIDEO = True

class VideoRecorder:
    def __init__(self, filename_arm, filename_fix):
        self.video_writer_arm = cv2.VideoWriter(filename_arm, cv2.VideoWriter_fourcc(*"MP4V"), 30, (640, 480))
        self.video_writer_fix = cv2.VideoWriter(filename_fix, cv2.VideoWriter_fourcc(*"MP4V"), 30, (640, 480))
        atexit.register(self.finish)

    def render(self, obs_arm, obs_fix):
        if obs_arm is not None and "im_rgbd" in obs_arm:
            self.video_writer_arm.write(cv2.cvtColor(np.asarray(obs_arm["im_rgbd"].color), cv2.COLOR_RGB2BGR))
        if obs_fix is not None and "im_rgbd" in obs_fix:
            self.video_writer_fix.write(cv2.cvtColor(np.asarray(obs_fix["im_rgbd"].color), cv2.COLOR_RGB2BGR))
    
    def finish(self):
        if self.video_writer_arm is not None:
            self.video_writer_arm.release()
            self.video_writer_arm = None
        if self.video_writer_fix is not None:
            self.video_writer_fix.release()
            self.video_writer_fix = None


if __name__ == "__main__":
    
    cfg = load_config()
    
    # basic configs
    devices = cfg.get("devices", [0])
    ckpt_path = cfg.get("inference_ckpt_path", "") or None
    model = cfg.get("model", "dit")
    image_size = int(cfg.get("image_size", 224)) # 224
    # horizon controls model x_seq_len and (by default) dataset Ta
    task = cfg.get("task", "<task>")
    horizon = int(cfg.get("horizon", cfg.get("Ta", 64)))
    # Ta controls how many steps to execute per inference (can be smaller than horizon)
    Ta = int(cfg.get("Ta", horizon))  # default: execute all predicted actions
    

    # load normalizer from config file
    normalizer_path = cfg.get("normalizer_path", None)
    if normalizer_path and os.path.exists(normalizer_path):
        with open(normalizer_path, "r") as f:
            normalizer_info = json.load(f)
        pos_max = np.array(normalizer_info["pos"]["max"], dtype=np.float32)
        pos_min = np.array(normalizer_info["pos"]["min"], dtype=np.float32)
        act_max = np.array(normalizer_info["action"]["max"], dtype=np.float32)
        act_min = np.array(normalizer_info["action"]["min"], dtype=np.float32)
        force_max = np.array(normalizer_info["force"]["max"], dtype=np.float32)
        force_min = np.array(normalizer_info["force"]["min"], dtype=np.float32)
        
        pos_normalizer = MinMaxNormalizer(X_max=pos_max, X_min=pos_min)
        action_normalizer = MinMaxNormalizer(X_max=act_max, X_min=act_min)
        force_normalizer = MinMaxNormalizer(X_max=force_max, X_min=force_min)
        
        cprint(f"   Load normalizer from config file: {normalizer_path}", "green")
        cprint(f"   Pos range: [{pos_min}, {pos_max}]", "green")
        cprint(f"   Action range: [{act_min}, {act_max}]", "green")
        cprint(f"   Force range: [{force_min}, {force_max}]", "green")
    else:
        raise ValueError(f"Normalizer file not found: {normalizer_path}")


    nn_diffusion = DiT1d(
        x_dim=13,                   # 13 dim action (6 dim position + 1 dim gripper action + 6 dim ext_force)
        x_seq_len=horizon,          # action horizon
        vec_emb_dim=256,            # lowdim 64*To 2 + force 128 = 128 + 128 = 256
        seq_emb_dim=512,            # image_feat_arm 256 + image_feat_fix 256 = 512  (with To)
        d_model=384, 
        n_heads=6,
        depth=12, 
        head_type="mlp",            
        use_cross_attn=True,
        adaLN_on_cross_attn=True,
        timestep_emb_type="untrainable_fourier",
        timestep_emb_params={"scale": 0.2},
        )

    # T_force from config
    T_force = int(cfg.get("T_force", 10))
    
    print(f"Inference config: horizon={horizon}, Ta={Ta}, To={cfg.get('To', 2)}, T_force={T_force}")
    print(f"Model will predict {horizon} steps, but only execute {Ta} steps per inference")
    
    nn_condition = MultiViewResnetWithLowdimObsSeqCondition(
            image_sz=image_size,
            in_channel=3,
            lowdim=7,                   # 6 dim position + 1 dim gripper state
            force_dim=6,                # 6 dim force
            T_force=T_force,            # 10 step force history
            image_emb_dim=256,
            lowdim_emb_dim=64,
            force_emb_dim=128,          # aggregated embedding dim for force history
            dropout=0.0,
        )

    policy = ContinuousRectifiedFlow(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition
    )
    
    device = f"cuda:{devices[0]}"


    NORM_PARAMS = (0.5, 0.5, 0.5)
    normalize = Normalize(NORM_PARAMS, NORM_PARAMS)
    resize = Resize((image_size, image_size))

    policy.load_state_dict(
        torch.load(f"{ckpt_path}", map_location="cpu", weights_only=False)[
            "state_dict"
        ]
    )
    policy = policy.to(device).eval()

    # Compute and print model parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    param_dtype = next(policy.parameters()).dtype
    bytes_per_param = param_dtype.itemsize
    
    print(f"Model loaded: {ckpt_path}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter data type: {param_dtype}")
    print(f"Model size: {total_params * bytes_per_param / 1024 / 1024:.2f} MB")

    print(f"{task}")



    # To=2 for image/state history, T_force=10 for force history
    To = int(cfg.get("To", 2))
    prior = torch.zeros((1, horizon, 13), device=device)  # 13 dim action (6 pos + 1 gripper + 6 ext_force)
    states = torch.zeros((1, To, 7), device=device)  # (batch, To, 7) To step state (6 dim pos + 1 dim gripper state)
    forces = torch.zeros((1, T_force, 6), device=device)  # (batch, T_force, 6) T_force step force history

    env = XarmEnv(action_mode="relative")
    rs_arm_env = RealsenseEnv(serial=None)  # set your arm-mounted camera serial
    rs_fix_env = RealsenseEnv(serial=None)  # set your fixed-view camera serial
    agent = SpacemouseAgent()
    keyboard_listener = KeyboardListener()
    
    video_dir = f"videos/{task}/{model}_horizon_{horizon}"
    os.makedirs(video_dir, exist_ok=True)
    if RECORD_VIDEO:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        recorder = VideoRecorder(
            f"{video_dir}/{timestamp}_arm.mp4",
            f"{video_dir}/{timestamp}_fix.mp4"
        )
        print(f"Recording videos:")
        print(f"  Arm view: {video_dir}/{timestamp}_arm.mp4")
        print(f"  Fix view: {video_dir}/{timestamp}_fix.mp4")
    else:
        recorder = None
    
    force_data_dir = f"force_data/{task}"
    os.makedirs(force_data_dir, exist_ok=True)
    
    # Find existing force data files and get the next number
    existing_files = [f for f in os.listdir(force_data_dir) if f.startswith("force_data_") and f.endswith(".txt")]
    if existing_files:
        # Extract numbers from filenames like "force_data_1.txt", "force_data_2.txt"
        numbers = []
        for f in existing_files:
            try:
                # Extract number between "force_data_" and ".txt"
                num_str = f.replace("force_data_", "").replace(".txt", "")
                # Try to parse as integer (skip timestamp-based filenames)
                if num_str.isdigit():
                    numbers.append(int(num_str))
            except:
                pass
        next_number = max(numbers) + 1 if numbers else 1
    else:
        next_number = 1
    
    force_data_filename = f"{force_data_dir}/force_data_{next_number}.txt"
    print(f"Force data will be saved to: {force_data_filename}")


    obs = env.reset()
    rs_arm_obs = rs_arm_env.reset()
    rs_fix_obs = rs_fix_env.reset()

    time.sleep(1)

    obs = env.step([0, 0, 0, 0, 0, 0], gripper_action=0, speed=100)
    rs_arm_obs = rs_arm_env.step()
    rs_fix_obs = rs_fix_env.step()
    print(f"Environment reset complete! Initial position: {obs['goal_pos'][:3]}")
    time.sleep(3)


    if RECORD_VIDEO:
        recorder.render(rs_arm_obs, rs_fix_obs)
    
    # wait for enter key to start inference, you can also adjust the gripper here
    goal_gripper_action = 0
    print("Waiting for enter key to start inference...")
    try:
        while True:
            action, buttons = agent.act()
            if buttons[0]:
                goal_gripper_action += 40
                if goal_gripper_action > 840:
                    goal_gripper_action = 840
            elif buttons[1]:
                goal_gripper_action -= 40
                if goal_gripper_action < 0:
                    goal_gripper_action = 0

            obs = env.step(action, goal_gripper_action, speed=100)
            rs_arm_obs = rs_arm_env.step()
            rs_fix_obs = rs_fix_env.step()
            
            key = keyboard_listener.get_key()
            if key == '\n':  # enter key
                keyboard_listener.stop_listening()
                cprint("Zeroing force sensor...", "yellow")
                if env.reset_force_sensor_zero():
                    cprint("Force sensor zeroed successfully!", "green")
                else:
                    cprint("Warning: Force sensor zero failed, but continuing...", "yellow")
                time.sleep(0.2)  # wait for sensor to settle
                break
            elif key == '\x03':  # Ctrl+C
                cprint("Inference cancelled", "red")
                keyboard_listener.stop_listening()
                sys.exit(0)
            time.sleep(0.1)
    except KeyboardInterrupt:
        keyboard_listener.stop_listening()
        sys.exit(0)
    finally:
        keyboard_listener.stop_listening()


    obs = env.step([0, 0, 0, 0, 0, 0], 0)
    rs_arm_obs = rs_arm_env.step()
    rs_fix_obs = rs_fix_env.step()
    if RECORD_VIDEO:
        recorder.render(rs_arm_obs, rs_fix_obs)
    
    obs_np = {
        "rgb_arm": np.asarray(rs_arm_obs["im_rgbd"].color),  # (480, 640, 3) uint8
        "rgb_fix": np.asarray(rs_fix_obs["im_rgbd"].color),  # (480, 640, 3) uint8
        "goal_pos": obs["goal_pos"],  # (6,) float64
        "gripper": obs["gripper_position"],
        "force": obs["ext_force"],  # (6,) float64
    }

    # state processing - (pos + gripper state)
    pos = obs_np["goal_pos"].astype(np.float32)  # 6 dim pos
    pos = pos_normalizer.normalize(pos) 
    gripper_position = obs_np["gripper"]
    gripper_state = 1.0 if gripper_position > 420 else 0.0  # binary gripper state
    pos_with_gripper = np.concatenate([pos, [gripper_state]], axis=0)  
    state = torch.tensor(pos_with_gripper, device=device, dtype=torch.float32) # 7 dim state
    
    # initialize: copy current state to all history steps
    for i in range(To):
        states[0, i] = state  # (batch, To, 7)
    
    # force data processing - get real force from obs
    force = obs_np["force"].astype(np.float32)  # 6 dim force
    force = force_normalizer.normalize(force)  # normalize
    force = torch.tensor(force, device=device, dtype=torch.float32)
    # initialize: copy current force to all history steps
    for i in range(T_force):
        forces[0, i] = force  # (batch, T_force, 6)

    # image processing - (rgb_arm + rgb_fix)
    rgb_arm = obs_np["rgb_arm"].astype(np.float32).transpose(2, 0, 1) / 255.0  # normalize to [0,1]
    rgb_arm = torch.tensor(rgb_arm, device=device, dtype=torch.float32)
    rgb_arm = resize(rgb_arm)  # resize to specified size
    rgb_arm = normalize(rgb_arm)  # normalize to [-1,1]
    rgb_arm = rgb_arm.unsqueeze(0).repeat(To, 1, 1, 1)  # (To, C, H, W)
    rgb_arm = rgb_arm[None]  # (batch, To, C, H, W) = (1, To, C, H, W)

    rgb_fix = obs_np["rgb_fix"].astype(np.float32).transpose(2, 0, 1) / 255.0  # normalize to [0,1]
    rgb_fix = torch.tensor(rgb_fix, device=device, dtype=torch.float32)
    rgb_fix = resize(rgb_fix)  # resize to specified size
    rgb_fix = normalize(rgb_fix)  # normalize to [-1,1]
    rgb_fix = rgb_fix.unsqueeze(0).repeat(To, 1, 1, 1)  # (To, C, H, W)
    rgb_fix = rgb_fix[None]  # (batch, To, C, H, W) = (1, To, C, H, W)
    
    try:
        chunk_count = 0
        while True:
            chunk_count += 1
            print(f"Start round {chunk_count} inference...")
            
            # condition input
            condition_cfg = {"image_arm": rgb_arm, "image_fix": rgb_fix, "lowdim": states, "force": forces}

            goal_states, log = policy.sample(
                prior,
                solver="euler",
                sample_steps=20,
                condition_cfg=condition_cfg,
                use_ema=False,
                w_cfg=1.0,
            )
            
            act = goal_states[0].cpu().numpy()  # (horizon, 13) - 6 dim delta_pose + 1 dim gripper + 6 dim ext_force
            
            # separate 13 dim action: 6 dim delta_pose + 1 dim gripper + 6 dim ext_force
            act_6d = act[:, :6]          # 6 dim delta_pose
            act_gripper = act[:, 6:7]    # 1 dim gripper
            act_ext_force = act[:, 7:]   # 6 dim predicted ext_force
            
            # unnormalize 6 dim delta_pose
            act_6d = action_normalizer.unnormalize(act_6d)
            
            # gripper action processing (binary, no unnormalize)
            act_gripper_binary = (act_gripper > 0.5).astype(np.float32)
            act_gripper_exec = np.where(act_gripper_binary > 0.5, 600, 0).astype(int)
            
            act_ext_force_raw = force_normalizer.unnormalize(act_ext_force)
            
            # execute action
            act_exec = act_6d
            gripper_exec = act_gripper_exec
            
            print(f"Predicted {horizon} steps, will execute {Ta} steps...")
            
            # control frequency
            control_freq = CONTROL_FREQ
            dt = 1.0 / control_freq
            
            for i in range(Ta):  # execute only the first Ta steps
                step_start_time = time.time()
                
  
                obs = env.step(act_exec[i], gripper_exec[i])
                rs_arm_obs = rs_arm_env.step()
                rs_fix_obs = rs_fix_env.step()
                print(f"Step {i}/{Ta} real_force_z: {obs['ext_force'][2]:.3f} predicted_force_z: {act_ext_force_raw[i][2]:.3f}")
                
                with open(force_data_filename, 'a') as f:
                    f.write(f"Chunk {chunk_count}, Step {i}: real_ext_force_z={obs['ext_force'][2]:.6f}, predicted_ext_force_z={act_ext_force_raw[i][2]:.6f}\n")
                
                if RECORD_VIDEO:
                    recorder.render(rs_arm_obs, rs_fix_obs)
                
                obs_np = {
                    "rgb_arm": np.asarray(rs_arm_obs["im_rgbd"].color),
                    "rgb_fix": np.asarray(rs_fix_obs["im_rgbd"].color),
                    "goal_pos": obs["goal_pos"],
                    "gripper": obs["gripper_position"],
                    "force": obs["ext_force"],
                }
                
                # update force history for the last T_force steps only
                if i >= max(0, Ta - T_force):
                    force = obs_np["force"].astype(np.float32)
                    force = force_normalizer.normalize(force)
                    force = torch.tensor(force, device=device, dtype=torch.float32)
                    force_idx = i - max(0, Ta - T_force)
                    forces[0, force_idx] = force

                # update obs history for the last To steps only
                if i >= max(0, Ta - To):
                    pos = obs_np["goal_pos"].astype(np.float32)
                    pos = pos_normalizer.normalize(pos)
                    gripper_position = obs_np["gripper"]
                    gripper_state = 1.0 if gripper_position > 420 else 0.0
                    pos_with_gripper = np.concatenate([pos, [gripper_state]], axis=0)
                    state = torch.tensor(pos_with_gripper, device=device, dtype=torch.float32)
                    obs_idx = i - max(0, Ta - To)
                    states[0, obs_idx] = state

                    # update image history in-place
                    new_image = obs_np["rgb_arm"].astype(np.float32).transpose(2, 0, 1) / 255.0
                    new_image = torch.tensor(new_image, device=device, dtype=torch.float32)
                    new_image = resize(new_image)
                    new_image = normalize(new_image)
                    rgb_arm[0, obs_idx] = new_image

                    new_image = obs_np["rgb_fix"].astype(np.float32).transpose(2, 0, 1) / 255.0
                    new_image = torch.tensor(new_image, device=device, dtype=torch.float32)
                    new_image = resize(new_image)
                    new_image = normalize(new_image)
                    rgb_fix[0, obs_idx] = new_image

                # enforce control frequency
                elapsed_time = time.time() - step_start_time
                sleep_time = dt - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        if RECORD_VIDEO:
            recorder.finish()

    except:
        print(traceback.format_exc())
        if RECORD_VIDEO:
            recorder.finish()

    finally:
        if RECORD_VIDEO:
            recorder.finish()
