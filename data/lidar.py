import numpy as np
import cv2

def sample_lidar_lines(
    depth_map: np.ndarray, intrinsics: np.ndarray, keep_ratio: float = 1.0) -> np.ndarray:
    v, u, _ = np.nonzero(depth_map)
    z = depth_map[v, u, 0]
    points = np.linalg.inv(intrinsics) @ (np.vstack([u, v, np.ones_like(u)]) * z)
    points = points.transpose([1, 0])

    scan_y = points[:, 1]
    distance = np.linalg.norm(points, 2, axis=1)
    pitch = np.arcsin(scan_y / distance)
    num_points = np.shape(pitch)[0]
    pitch = np.reshape(pitch, (num_points, 1))
    max_pitch = np.max(pitch)
    min_pitch = np.min(pitch)
    angle_interval = (max_pitch - min_pitch) / 65.0 
    angle_label = np.round((pitch - min_pitch) / angle_interval)
    angle_label+= (1.0 / keep_ratio)/2.0 
    sampling_mask = angle_label % (1.0 / keep_ratio) == 0
    

    final_mask = np.zeros_like(depth_map, dtype=bool)
    final_mask[depth_map[..., 0] > 0] = sampling_mask
    sampled_depth = np.zeros_like(final_mask, dtype=np.float32)
    sampled_depth[final_mask] = depth_map[final_mask]
    return sampled_depth