import torch


def compute_joint_positions(
    root_position: torch.Tensor,  # 2
    angles: torch.Tensor,  # n_joints
    lengths: torch.Tensor,  # n_joints
    parents: torch.Tensor,  # n_joints
) -> torch.Tensor:  # n_joints x 2
    positions = torch.zeros((angles.shape[0], 2))

    positions[0] = root_position

    for i in range(1, angles.shape[0]):
        parent_position = positions[parents[i]]
        # rotation_mat = torch.Tensor([
        #     [torch.cos(angles[i]), -torch.sin(angles[i])],
        #     [torch.sin(angles[i]), torch.cos(angles[i])]
        # ])
        # rotated_pos = rotation_mat @ torch.Tensor([lengths[i], 0])
        rotated_pos = torch.Tensor(
            [torch.cos(angles[i]) * lengths[i], torch.sin(angles[i]) * lengths[i]])
        positions[i] = parent_position + rotated_pos

    return positions


def ik_jacobian(
    parents: torch.Tensor,  # n_joints
    angles: torch.Tensor,  # n_joints
    lengths: torch.Tensor,  # n_joints
) -> torch.Tensor:  # n_coordinates x n_parameters
    n_coordinates = 2 * angles.shape[0]  # 2n
    n_parameters = angles.shape[0] + 1  # n+1
    jacobian = torch.zeros((n_coordinates, n_parameters))

    for i in range(angles.shape[0]):
        jacobian[2*i, 0] = 1  # dxi/dx0
        jacobian[2*i+1, 1] = 1  # dyi/dy0

    for i in range(1, angles.shape[0]):
        j = i
        while j != 0:
            jacobian[2*i, j+1]   = -torch.sin(angles[j]) * lengths[j]
            jacobian[2*i+1, j+1] = torch.cos(angles[j]) * lengths[j]
            j = parents[j]

    return jacobian


def ik_shift(
    current_positions: torch.Tensor,  # n_joints x 2
    target_positions: torch.Tensor,  # n_joints x 2
) -> torch.Tensor:  # n_coordinates

    shift = target_positions - current_positions
    return shift.flatten()


def ik_solve(
    jacobian: torch.Tensor,  # n_coordinates x n_parameters
    shift: torch.Tensor,  # n_coordinates
) -> torch.Tensor:  # n_parameters
    return torch.linalg.lstsq(jacobian, shift)[0]


def apply_changes(
    positions: torch.Tensor,  # n_joints x 2
    angles: torch.Tensor,  # n_joints
    changes: torch.Tensor,  # n_parameters
    parents: torch.Tensor,  # n_joints
    lengths: torch.Tensor,  # n_joints
) -> torch.Tensor:  # (n_joints x 2, n_joints)

    positions[0, 0] += changes[0]
    positions[0, 1] += changes[1]
    angles[1:] += changes[2:]
    positions = compute_joint_positions(positions[0], angles, lengths, parents)

    return positions, angles
