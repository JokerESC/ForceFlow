import os
import sys
import json
import numpy as np
import zarr
from termcolor import cprint


DATASET_PATH = "data/<task>.zarr"  # set to your dataset path


# jpeg codec required for zarr to read jpeg-compressed rgb data
try:
    from CleanDiffuser.image_codecs.jpeg import jpeg  # noqa
except ImportError:
    cprint("jpeg codec not found - RGB data may fail to load", "yellow")


def calculate_normalizer_params(dataset_path):
    try:
        dataset = zarr.open(dataset_path, 'r')
        data_group = dataset['data']

        cprint("Computing normalizer params...", "cyan")
        normalizer_info = {}

        if 'pos' in data_group:
            pos_data = data_group['pos'][:]
            pos_max = np.max(pos_data, axis=0)
            pos_min = np.min(pos_data, axis=0)
            normalizer_info['pos'] = {
                'max': pos_max.tolist(),
                'min': pos_min.tolist(),
                'shape': pos_data.shape,
                'dtype': str(pos_data.dtype)
            }
            cprint("pos statistics:", "green")
            cprint(f"  shape: {pos_data.shape}", "cyan")
            cprint(f"  max: {pos_max}", "cyan")
            cprint(f"  min: {pos_min}", "cyan")

        if 'action' in data_group:
            action_data = data_group['action'][:]
            action_max = np.max(action_data, axis=0)
            action_min = np.min(action_data, axis=0)
            normalizer_info['action'] = {
                'max': action_max.tolist(),
                'min': action_min.tolist(),
                'shape': action_data.shape,
                'dtype': str(action_data.dtype)
            }
            cprint("action statistics:", "green")
            cprint(f"  shape: {action_data.shape}", "cyan")
            cprint(f"  max: {action_max}", "cyan")
            cprint(f"  min: {action_min}", "cyan")

        if 'force' in data_group:
            force_data = data_group['force'][:]
            force_max = np.max(force_data, axis=0)
            force_min = np.min(force_data, axis=0)
            normalizer_info['force'] = {
                'max': force_max.tolist(),
                'min': force_min.tolist(),
                'shape': force_data.shape,
                'dtype': str(force_data.dtype)
            }
            cprint("force statistics:", "green")
            cprint(f"  shape: {force_data.shape}", "cyan")
            cprint(f"  max: {force_max}", "cyan")
            cprint(f"  min: {force_min}", "cyan")

        return normalizer_info

    except Exception as e:
        cprint(f"Failed to compute normalizer params: {e}", "red")
        return None


def fix_episode_ends(dataset_path):
    """Recompute episode_ends from the episode field and write to dataset."""
    try:
        dataset = zarr.open(dataset_path, 'a')
        data_group = dataset['data']
        meta_group = dataset['meta']

        if 'episode' not in data_group:
            cprint("Missing 'episode' field, cannot recompute episode_ends", "red")
            return False

        episode_ids = data_group['episode'][:]
        total_steps = len(episode_ids)

        cprint("Recomputing episode_ends...", "cyan")
        cprint(f"  total steps: {total_steps}", "cyan")

        transitions = []
        current_id = episode_ids[0]
        for i in range(1, len(episode_ids)):
            if episode_ids[i] != current_id:
                transitions.append(i)
                current_id = episode_ids[i]
        # include the final boundary
        transitions.append(len(episode_ids))
        correct_episode_ends = np.array(transitions, dtype=np.uint32)

        current_episode_ends = None
        if 'episode_ends' in meta_group:
            current_episode_ends = meta_group['episode_ends'][:]
            cprint(f"  current: {len(current_episode_ends)} episodes", "cyan")
        else:
            cprint("  no episode_ends found, will create", "cyan")

        cprint(f"  recomputed: {len(correct_episode_ends)} episodes", "cyan")

        needs_fix = False
        if current_episode_ends is None:
            needs_fix = True
            cprint("  reason: episode_ends missing", "yellow")
        elif len(current_episode_ends) != len(correct_episode_ends):
            needs_fix = True
            cprint(f"  reason: episode count mismatch ({len(current_episode_ends)} vs {len(correct_episode_ends)})", "yellow")
        elif not np.array_equal(current_episode_ends, correct_episode_ends):
            needs_fix = True
            cprint("  reason: episode boundaries differ", "yellow")

        if needs_fix:
            if 'episode_ends' in meta_group:
                del meta_group['episode_ends']
            meta_group.create_dataset('episode_ends', data=correct_episode_ends, dtype=np.uint32)
            cprint("episode_ends updated", "green")
            cprint(f"  new episode_ends: {correct_episode_ends}", "cyan")
            if len(correct_episode_ends) > 1:
                lengths = np.diff(np.concatenate([[0], correct_episode_ends]))
                cprint(f"  episode lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}", "cyan")
        else:
            cprint("episode_ends already correct", "green")

        return True

    except Exception as e:
        cprint(f"Failed to update episode_ends: {e}", "red")
        import traceback
        cprint(f"Traceback:\n{traceback.format_exc()}", "red")
        return False


def build_normalizer_json(normalizer_info):
    """Build the dict to save as normalizer JSON, keeping only min/max statistics."""
    result = {}
    if 'pos' in normalizer_info:
        result['pos'] = {'max': normalizer_info['pos']['max'], 'min': normalizer_info['pos']['min']}
    if 'action' in normalizer_info:
        result['action'] = {'max': normalizer_info['action']['max'], 'min': normalizer_info['action']['min']}
    if 'force' in normalizer_info:
        result['force'] = {'max': normalizer_info['force']['max'], 'min': normalizer_info['force']['min']}
    return result


def save_normalizer_info(dataset_path, normalizer_info):
    """Save normalizer statistics to a JSON file compatible with xArmDataset."""
    try:
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        json_file = f"{dataset_path}/{base_name}_normalizer.json"
        json_payload = build_normalizer_json(normalizer_info)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False)
        cprint(f"Saved normalizer JSON: {json_file}", "green")
        return json_file
    except Exception as e:
        cprint(f"Failed to save normalizer JSON: {e}", "red")
        return None


def validate_dataset_structure(dataset_path):
    """Validate dataset group structure and field shapes/dtypes."""
    try:
        dataset = zarr.open(dataset_path, 'r')
        cprint(f"Opened dataset: {dataset_path}", "green")

        if 'data' not in dataset:
            cprint("Missing 'data' group", "red")
            return False
        if 'meta' not in dataset:
            cprint("Missing 'meta' group", "red")
            return False

        data_group = dataset['data']

        required_datasets = {
            'rgb_fix':       {'shape': (None, 3, 240, 320), 'dtype': np.uint8},
            'rgb_arm':       {'shape': (None, 3, 240, 320), 'dtype': np.uint8},
            'pos':           {'shape': (None, 6),           'dtype': np.float32},
            'force':         {'shape': (None, 6),           'dtype': np.float32},  # force sensor data
            'action':        {'shape': (None, 6),           'dtype': np.float32},  # 6-dim action, no gripper
            'gripper_state': {'shape': (None, 1),           'dtype': np.float32},  # binary gripper state
            'gripper_action':{'shape': (None, 1),           'dtype': np.float32},  # binary gripper action
            'episode':       {'shape': (None,),             'dtype': np.uint16},
        }

        for dataset_name, expected in required_datasets.items():
            if dataset_name not in data_group:
                cprint(f"Missing dataset: {dataset_name}", "red")
                return False

            ds = data_group[dataset_name]
            actual_shape = ds.shape
            actual_dtype = ds.dtype

            if len(actual_shape) != len(expected['shape']):
                cprint(f"{dataset_name}: rank mismatch (expected {len(expected['shape'])}, got {len(actual_shape)})", "red")
                return False

            for i in range(1, len(expected['shape'])):
                if actual_shape[i] != expected['shape'][i]:
                    cprint(f"{dataset_name}: shape mismatch (expected {expected['shape']}, got {actual_shape})", "red")
                    return False

            if actual_dtype != expected['dtype']:
                cprint(f"{dataset_name}: dtype mismatch (expected {expected['dtype']}, got {actual_dtype})", "red")
                return False

            cprint(f"{dataset_name}: shape={actual_shape} dtype={actual_dtype}", "green")

        return True

    except Exception as e:
        cprint(f"Failed to open dataset: {e}", "red")
        return False


def validate_data_consistency(dataset_path):
    """Check that all data fields have the same number of steps."""
    try:
        dataset = zarr.open(dataset_path, 'r')
        data_group = dataset['data']

        lengths = {key: data_group[key].shape[0] for key in data_group.keys()}

        cprint("Step counts:", "cyan")
        for key, length in lengths.items():
            cprint(f"  {key}: {length}", "cyan")

        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            cprint(f"Inconsistent lengths: {lengths}", "red")
            return False
        else:
            main_length = list(unique_lengths)[0]
            cprint(f"All fields consistent: {main_length} steps", "green")

        return True

    except Exception as e:
        cprint(f"Failed to check consistency: {e}", "red")
        return False


def validate_data_ranges(dataset_path):
    """Validate value ranges for rgb, gripper state/action, and episode fields."""
    try:
        dataset = zarr.open(dataset_path, 'r')
        data_group = dataset['data']

        cprint("Data range check:", "cyan")

        if 'rgb_fix' in data_group:
            try:
                rgb_data = data_group['rgb_fix'][:]
                if rgb_data.min() < 0 or rgb_data.max() > 255:
                    cprint(f"rgb_fix out of range: [{rgb_data.min()}, {rgb_data.max()}]", "red")
                    return False
                cprint(f"rgb_fix range: [{rgb_data.min()}, {rgb_data.max()}]", "green")
            except Exception as e:
                cprint(f"rgb_fix read failed: {e}", "yellow")

        if 'rgb_arm' in data_group:
            try:
                rgb_data = data_group['rgb_arm'][:]
                if rgb_data.min() < 0 or rgb_data.max() > 255:
                    cprint(f"rgb_arm out of range: [{rgb_data.min()}, {rgb_data.max()}]", "red")
                    return False
                cprint(f"rgb_arm range: [{rgb_data.min()}, {rgb_data.max()}]", "green")
            except Exception as e:
                cprint(f"rgb_arm read failed: {e}", "yellow")

        if 'gripper_state' in data_group:
            try:
                gripper_state_data = data_group['gripper_state'][:]
                unique_states = np.unique(gripper_state_data)
                state_counts = {state: np.sum(gripper_state_data == state) for state in unique_states}
                if not all(state in [0.0, 1.0] for state in unique_states):
                    cprint(f"gripper_state has non-binary values: {unique_states}", "yellow")
                cprint("gripper_state:", "green")
                cprint(f"  unique values: {unique_states}", "cyan")
                for state, count in state_counts.items():
                    percentage = (count / len(gripper_state_data)) * 100
                    state_name = "closed" if state == 0.0 else "open"
                    cprint(f"  {state_name}({state}): {count} steps ({percentage:.1f}%)", "cyan")
            except Exception as e:
                cprint(f"gripper_state read failed: {e}", "yellow")

        if 'gripper_action' in data_group:
            try:
                gripper_action_data = data_group['gripper_action'][:]
                unique_actions = np.unique(gripper_action_data)
                action_counts = {action: np.sum(gripper_action_data == action) for action in unique_actions}
                if not all(action in [0.0, 1.0] for action in unique_actions):
                    cprint(f"gripper_action has non-binary values: {unique_actions}", "yellow")
                cprint("gripper_action:", "green")
                cprint(f"  unique values: {unique_actions}", "cyan")
                for action, count in action_counts.items():
                    percentage = (count / len(gripper_action_data)) * 100
                    action_name = "close" if action == 0.0 else "open"
                    cprint(f"  {action_name}({action}): {count} steps ({percentage:.1f}%)", "cyan")
            except Exception as e:
                cprint(f"gripper_action read failed: {e}", "yellow")

        if 'episode' in data_group:
            try:
                episode_data = data_group['episode'][:]
                unique_episodes = np.unique(episode_data)
                cprint(f"Episode ids: {unique_episodes}", "green")
            except Exception as e:
                cprint(f"episode field read failed: {e}", "yellow")

        return True

    except Exception as e:
        cprint(f"Data range check failed: {e}", "red")
        return False


def validate_episode_structure(dataset_path):
    """Print per-episode step counts."""
    try:
        dataset = zarr.open(dataset_path, 'r')
        data_group = dataset['data']

        if 'episode' not in data_group:
            cprint("No episode field, skipping structure check", "yellow")
            return True

        try:
            episode_data = data_group['episode'][:]
            unique_episodes = np.unique(episode_data)
            cprint("Episode structure:", "cyan")
            cprint(f"  total episodes: {len(unique_episodes)}", "cyan")
            for episode_id in unique_episodes:
                episode_length = np.sum(episode_data == episode_id)
                cprint(f"  episode {episode_id}: {episode_length} steps", "cyan")
        except Exception as e:
            cprint(f"episode structure analysis failed: {e}", "yellow")

        return True

    except Exception as e:
        cprint(f"Episode structure validation failed: {e}", "red")
        return False


def main():
    dataset_path = DATASET_PATH

    if dataset_path is None:
        raise ValueError("dataset_path is not set")

    cprint("Validating dataset...", "yellow")
    cprint("=" * 50, "cyan")
    cprint(f"Dataset: {dataset_path}", "cyan")

    cprint("\n1. Checking dataset structure...", "yellow")
    if not validate_dataset_structure(dataset_path):
        return

    cprint("\n2. Checking data consistency...", "yellow")
    if not validate_data_consistency(dataset_path):
        return

    cprint("\n3. Checking data ranges...", "yellow")
    if not validate_data_ranges(dataset_path):
        cprint("Data range check failed", "red")
        return

    cprint("\n4. Checking episode structure...", "yellow")
    if not validate_episode_structure(dataset_path):
        return

    cprint("\n5. Fixing episode_ends...", "yellow")
    if not fix_episode_ends(dataset_path):
        cprint("episode_ends fix failed", "red")
        return

    cprint("\n6. Computing normalizer params...", "yellow")
    normalizer_info = calculate_normalizer_params(dataset_path)
    if normalizer_info:
        json_file = save_normalizer_info(dataset_path, normalizer_info)
        if json_file:
            cprint(f"Generated file: {json_file}", "green")

    cprint("\n" + "=" * 50, "green")
    cprint("All checks passed!", "green")
    cprint("=" * 50, "green")


if __name__ == "__main__":
    main()
