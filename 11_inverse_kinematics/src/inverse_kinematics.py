import torch


def compute_joint_positions(
    root_position: torch.Tensor,  # 2
    angles: torch.Tensor,  # n_joints
    lengths: torch.Tensor,  # n_joints
    parents: torch.Tensor,  # n_joints
) -> torch.Tensor:  # n_joints x 2
    positions = torch.zeros((angles.shape[0], 2))

    # TODO implement ...

    return positions


def ik_jacobian(
    parents: torch.Tensor,  # n_joints
    angles: torch.Tensor,  # n_joints
    lengths: torch.Tensor,  # n_joints
) -> torch.Tensor:  # n_coordinates x n_parameters
    n_coordinates = 2 * angles.shape[0]  # 2n
    n_parameters = angles.shape[0] + 1  # n+1
    jacobian = torch.zeros((n_coordinates, n_parameters))

    # TODO implement ...

    return jacobian


def ik_shift(
    current_positions: torch.Tensor,  # n_joints x 2
    target_positions: torch.Tensor,  # n_joints x 2
) -> torch.Tensor:  # n_coordinates
    n_coordinates = target_positions.shape[0] * target_positions.shape[1]  # 2n
    shift = torch.zeros(n_coordinates)

    # TODO implement ...

    return shift


def ik_solve(
    jacobian: torch.Tensor,  # n_coordinates x n_parameters
    shift: torch.Tensor,  # n_coordinates
) -> torch.Tensor:  # n_parameters
    parameter_changes = torch.zeros((jacobian.shape[1]))

    # TODO implement ...

    return parameter_changes


def apply_changes(
    positions: torch.Tensor,  # n_joints x 2
    angles: torch.Tensor,  # n_joints
    changes: torch.Tensor,  # n_parameters
    parents: torch.Tensor,  # n_joints
    lengths: torch.Tensor,  # n_joints
) -> torch.Tensor:  # (n_joints x 2, n_joints)
    # TODO implement ...

    return positions, angles
