"""Where the hand is, from the arm's 7 joint angles, for many states at once (forward kinematics in
torch). Planning with a world model needs it: the model predicts joint angles, and success depends on
how far the hand is from the nut. It walks the scene's chain of bodies from the world to the hand
site the way MuJoCo does (test_train.py checks it against MuJoCo).
"""

import mujoco
import numpy as np
import torch


def rotation(quaternions):
    """[..., 3, 3] rotation matrices from (w, x, y, z) quaternions."""
    w, x, y, z = quaternions.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1).view(*w.shape, 3, 3)


class HandFinder:
    def __init__(self, model, site="gripper0_grip_site", dtype=torch.float64, device="cpu"):
        as_tensor = lambda values: torch.tensor(np.array(values), dtype=dtype, device=device)
        site_id = model.site(site).id
        chain, body = [], model.site_bodyid[site_id]
        while body != 0:
            chain.append(body)
            body = model.body_parentid[body]
        self.links = []                 # from the world outward: (position, orientation, joint or None)
        for body in reversed(chain):
            joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] == body]
            assert len(joints) <= 1 and all(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in joints)
            joint = None
            if joints:
                j = joints[0]
                joint = (int(model.jnt_qposadr[j]), as_tensor(model.jnt_pos[j]), as_tensor(model.jnt_axis[j]))
            self.links.append((as_tensor(model.body_pos[body]), rotation(as_tensor(model.body_quat[body])), joint))
        self.site_position = as_tensor(model.site_pos[site_id])

    def hand(self, joint_angles):
        """[n, 3]: the hand site's position for joint angles [n, 7] (the arm's qpos)."""
        n = len(joint_angles)
        position = torch.zeros(n, 3, dtype=joint_angles.dtype, device=joint_angles.device)
        orientation = torch.eye(3, dtype=joint_angles.dtype, device=joint_angles.device).expand(n, 3, 3)
        for offset, turn, joint in self.links:
            position = position + orientation @ offset
            orientation = orientation @ turn
            if joint is not None:
                index, anchor, axis = joint
                anchor_world = position + orientation @ anchor
                angle = joint_angles[:, index]
                half = (angle / 2)[:, None]
                spin = rotation(torch.cat([half.cos(), axis * half.sin()], dim=1))
                orientation = orientation @ spin
                position = anchor_world - orientation @ anchor
        return position + orientation @ self.site_position
