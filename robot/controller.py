"""robosuite's controller for many copies of the world at once, in torch, for the GPU.

The same math as sim.Sim (set_goal, torques, the finger targets), but on batches: every input has a
first dimension for the copies. From the physics it reads, for each copy:
    hand          the hand's position [3] and orientation [3, 3]
    anchors, axes where each of the arm's 7 joints is and which way it turns [7, 3] (the hand's
                  Jacobians follow from them: every arm joint is a hinge between the base and the hand)
    mass          the arm's mass matrix [7, 7]
    holding_up    the forces that hold the arm up against gravity and motion [7]
    q, v          the arm's joint angles and speeds [7]
Both MuJoCo on the CPU and MuJoCo Warp on the GPU give these, so the same code runs on both.
test_sim.py checks that it gives sim.Sim's motor commands.

Where robosuite (and sim.Sim) take pseudo-inverses of the small "how heavy the hand feels" matrices,
this takes plain inverses: the two agree except at the arm's singular poses, and a plain inverse that
does not stop to check for errors lets the GPU record the whole step as one graph and replay it.
"""

import torch

from sim import DAMPING, FINGER_SPEED, HOME, HOME_DAMPING, HOME_STIFFNESS, LARGEST, STIFFNESS

_constants = {}


def constant(name, values, like):
    """A fixed tensor, made once per device and number type (making one while the GPU records a
    graph is not allowed)."""
    key = (name, like.device, like.dtype)
    if key not in _constants:
        _constants[key] = torch.as_tensor(values, dtype=like.dtype, device=like.device)
    return _constants[key]


def jacobians(hand_position, anchors, axes):
    """[copies, 3, 7] each: how the hand moves, and turns, as each arm joint turns."""
    moves = torch.cross(axes, hand_position[:, None, :] - anchors, dim=-1)
    return moves.transpose(1, 2), axes.transpose(1, 2)


def rotation(rotvec):
    """[copies, 3, 3] rotation matrices for rotations given as axis times angle (Rodrigues' formula)."""
    angle = rotvec.norm(dim=-1, keepdim=True)
    axis = rotvec / angle.clamp_min(1e-12)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    cross = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).view(-1, 3, 3)
    eye = constant("eye3", torch.eye(3), rotvec).expand_as(cross)
    sin, cos = angle.sin()[..., None], angle.cos()[..., None]
    return eye + sin * cross + (1 - cos) * cross @ cross


def goal(action, hand_position, hand_orientation):
    """The control step's goal for the hand: the action's first 3 numbers move it, the next 3 turn it."""
    delta = action[:, :6].clamp(-1, 1) * constant("largest", LARGEST, action)
    return hand_position + delta[:, :3], rotation(delta[:, 3:]) @ hand_orientation


def invert(matrices):
    """Batched inverses, without stopping to check for singular matrices (which would make the GPU wait)."""
    return torch.linalg.inv_ex(matrices)[0]


def torques(goal_position, goal_orientation, hand_position, hand_orientation, anchors, axes, mass, holding_up, q, v,
            limits):
    """[copies, 7] arm torques, as sim.Sim.torques works them out, clipped to the motors' limits [7, 2]."""
    jac_position, jac_turn = jacobians(hand_position, anchors, axes)
    velocity, turning = (jac_position @ v[..., None])[..., 0], (jac_turn @ v[..., None])[..., 0]
    turn_error = 0.5 * torch.cross(hand_orientation, goal_orientation, dim=1).sum(dim=-1)
    force = STIFFNESS * (goal_position - hand_position) - DAMPING * velocity
    twist = STIFFNESS * turn_error - DAMPING * turning
    jac = torch.cat([jac_position, jac_turn], dim=1)
    mass_inverse = invert(mass)
    heavy_position = invert(jac_position @ mass_inverse @ jac_position.transpose(1, 2))
    heavy_turn = invert(jac_turn @ mass_inverse @ jac_turn.transpose(1, 2))
    heavy = invert(jac @ mass_inverse @ jac.transpose(1, 2))
    wrench = torch.cat([heavy_position @ force[..., None], heavy_turn @ twist[..., None]], dim=1)
    result = (jac.transpose(1, 2) @ wrench)[..., 0] + holding_up
    nullspace = constant("eye7", torch.eye(7), q) - mass_inverse @ jac.transpose(1, 2) @ heavy @ jac
    drift = mass @ (HOME_STIFFNESS * (constant("home", HOME, q) - q) - HOME_DAMPING * v)[..., None]
    result = result + (nullspace.transpose(1, 2) @ drift)[..., 0]
    return result.clamp(limits[:, 0], limits[:, 1])


def finger_targets(targets, gripper):
    """One physics step's change of the finger targets [copies, 2], by the sign of the gripper action."""
    direction = constant("finger direction", [-1.0, 1.0], targets)
    return (targets + direction * FINGER_SPEED * torch.sign(gripper)[:, None]).clamp(-1, 1)
