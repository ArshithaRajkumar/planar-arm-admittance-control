"""
kinematics.py — Forward Kinematics, Jacobian, and Board-Frame Utilities
=========================================================================
Chopstick Crane: Planar 3-link arm in the x-z plane.

Mathematical derivation
-----------------------
All joints rotate about the y-axis (out of the x-z plane).
Cumulative angles: α₁ = θ₁,  α₂ = θ₁+θ₂,  α₃ = θ₁+θ₂+θ₃

Pen-tip position (world frame):
    pₓ = L1·cos(α₁) + L2·cos(α₂) + L3·cos(α₃)
    pz = L1·sin(α₁) + L2·sin(α₂) + L3·sin(α₃)

Jacobian (2×3):
    J[0, i] = ∂pₓ/∂θᵢ = −Σⱼ≥ᵢ Lⱼ·sin(αⱼ)
    J[1, i] = ∂pz/∂θᵢ = +Σⱼ≥ᵢ Lⱼ·cos(αⱼ)

Board frame
-----------
The board pivots about a fixed hinge at BOARD_HINGE_POS = (0.50, -0.10) m.
When tilt angle φ = 0, the board surface points in the +x direction.
Board normal (pointing away from surface, i.e. upward at φ=0):
    n = [−sin(φ), cos(φ)]    (in x-z frame)
Board tangent (along the surface):
    t = [ cos(φ), sin(φ)]

Target curve on the board (board frame → world frame):
    u(s) = SWEEP_OFFSET + SWEEP_AMPLITUDE · sin(2π·s),   s ∈ [0, 1]
    p_world(s, φ) = BOARD_HINGE_POS + u(s)·t(φ)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Physical parameters
# ─────────────────────────────────────────────────────────────────────────────

# Link lengths [m]
L1, L2, L3   = 0.30, 0.25, 0.15
LINK_LENGTHS  = np.array([L1, L2, L3])

# Arm base position (world frame) — matches model.xml
ARM_BASE_POS  = np.array([0.0, 0.0])          # (x, z)

# Board hinge position (world x-z frame) — matches model.xml
BOARD_HINGE_POS = np.array([0.45, 0.0])     # (x, z)

# Board passive dynamics
BOARD_K        = 3.0      # spring stiffness  [N·m/rad]
BOARD_B        = 0.3      # damping coeff.    [N·m·s/rad]
BOARD_INERTIA  = 0.02     # moment of inertia [kg·m²]

# Contact-force target band [N]
F_MIN, F_MAX   = 1.5, 6.0
F_TARGET       = 0.5 * (F_MIN + F_MAX)

# Sinusoidal sweep parameters
SWEEP_OFFSET    = 0.10    # [m] mean position along board from hinge
SWEEP_AMPLITUDE = 0.08    # [m] half-amplitude of sinusoidal excursion

# Joint limits [rad]
J_LIMITS = np.array([
    [-np.pi,      np.pi   ],   # j1
    [-2.618,      2.618   ],   # j2
    [-np.pi,      np.pi   ],   # j3
])


# ─────────────────────────────────────────────────────────────────────────────
# Forward Kinematics
# ─────────────────────────────────────────────────────────────────────────────

def forward_kinematics(theta: np.ndarray) -> np.ndarray:
    """
    Pen-tip position in the world (x-z) frame.

    Parameters
    ----------
    theta : array-like, shape (3,)
        Joint angles [θ₁, θ₂, θ₃] in radians.

    Returns
    -------
    p : ndarray, shape (2,)
        [pₓ, pz] in metres.
    """
    t     = np.asarray(theta, dtype=float)
    alpha = np.cumsum(t)                      # cumulative angles
    px    = np.dot(LINK_LENGTHS, np.cos(alpha))
    # MuJoCo Ry(θ) sends +x toward -z for positive θ (right-hand rule about +y).
    # So pz = -sin(α) to match MuJoCo's world frame where +θ tilts arm downward.
    pz    = -np.dot(LINK_LENGTHS, np.sin(alpha))
    return np.array([px, pz])


def all_joint_positions(theta: np.ndarray) -> np.ndarray:
    """
    World-frame (x, z) positions of the base, each joint, and the pen tip.

    Returns
    -------
    pts : ndarray, shape (5, 2)
        Rows: [base, j1_end/j2_start, j2_end/j3_start, j3_end/tip_start, pen_tip]
              (base and j1 coincide for a fixed base)
    """
    t     = np.asarray(theta, dtype=float)
    alpha = np.cumsum(t)
    pts   = np.zeros((5, 2))
    pts[0] = ARM_BASE_POS
    for i, (l, a) in enumerate(zip(LINK_LENGTHS, alpha)):
        # pz = -sin(α) to match MuJoCo's Ry convention
        pts[i + 1] = pts[i] + l * np.array([np.cos(a), -np.sin(a)])
    return pts[:4]   # base, joint2 pos, joint3 pos, pen_tip


# ─────────────────────────────────────────────────────────────────────────────
# Jacobian
# ─────────────────────────────────────────────────────────────────────────────

def jacobian(theta: np.ndarray) -> np.ndarray:
    """
    Analytical 2×3 Jacobian: J[i, j] = ∂p_i / ∂θ_j.

    Parameters
    ----------
    theta : array-like, shape (3,)

    Returns
    -------
    J : ndarray, shape (2, 3)
    """
    t     = np.asarray(theta, dtype=float)
    alpha = np.cumsum(t)
    J     = np.zeros((2, 3))
    for i in range(3):
        # Joint θᵢ affects links i, i+1, … (2)
        J[0, i] = -np.dot(LINK_LENGTHS[i:], np.sin(alpha[i:]))  # ∂pₓ/∂θᵢ
        # Negated pz means ∂pz/∂θᵢ = -∂(+sin)/∂θᵢ = +sin (double negative → positive dot)
        J[1, i] = -np.dot(LINK_LENGTHS[i:], np.cos(alpha[i:]))  # ∂pz/∂θᵢ  (pz=-sin → J=-cos)
    return J


def numerical_jacobian(theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Finite-difference Jacobian (for testing analytical J).
    """
    J_num = np.zeros((2, 3))
    for i in range(3):
        tp, tm  = theta.copy(), theta.copy()
        tp[i]  += eps;  tm[i] -= eps
        J_num[:, i] = (forward_kinematics(tp) - forward_kinematics(tm)) / (2 * eps)
    return J_num


# ─────────────────────────────────────────────────────────────────────────────
# Board-frame utilities
# ─────────────────────────────────────────────────────────────────────────────

def board_tangent(phi: float) -> np.ndarray:
    """
    Unit tangent vector along board surface (x-z frame).
    Vertical board: At φ=0 → [0, 1] (board points in +z).
    t = [sin(φ), cos(φ)]
    """
    return np.array([np.sin(phi), np.cos(phi)])


def board_normal(phi: float) -> np.ndarray:
    """
    Unit normal vector pointing AWAY from board surface (towards the robot).
    Vertical board: At φ=0, board surface points left (-x).
    n = [-cos(φ), sin(φ)]
    """
    return np.array([-np.cos(phi), np.sin(phi)])


def board_to_world(s_coord: float, phi: float) -> np.ndarray:
    """
    Map a distance s_coord along the board (from hinge) to world (x, z).
    """
    return BOARD_HINGE_POS + s_coord * board_tangent(phi)


def target_curve(s: float, phi: float) -> np.ndarray:
    """
    Desired pen-tip position (ON the board surface) for sweep parameter s ∈ [0,1].

    Prescribed curve in board-frame arc-length:
        u(s) = SWEEP_OFFSET + SWEEP_AMPLITUDE · sin(2π·s)

    Parameters
    ----------
    s   : float   Sweep parameter ∈ [0, 1].
    phi : float   Current board tilt [rad].

    Returns
    -------
    p_target : ndarray, shape (2,)
    """
    u = SWEEP_OFFSET + SWEEP_AMPLITUDE * np.sin(2 * np.pi * s)
    # The board surface is offset from the hinge by its half-thickness (0.020m).
    # The pen sphere has a radius of 0.009m.
    # To just 'kiss' the board, the pen center must be offset by 0.029m in the normal direction.
    return BOARD_HINGE_POS + u * board_tangent(phi) + 0.029 * board_normal(phi)


def target_with_offset(s: float, phi: float, normal_offset: float = 0.0) -> np.ndarray:
    """
    Target pen-tip position with a normal offset for force regulation.

    positive normal_offset → target is INSIDE the board → pen pushes harder.
    negative normal_offset → target is ABOVE the board → pen releases.

    Parameters
    ----------
    s             : float   Sweep parameter ∈ [0, 1].
    phi           : float   Current board tilt [rad].
    normal_offset : float   Signed penetration offset [m].
    """
    p_surface = target_curve(s, phi)
    n         = board_normal(phi)
    # Offset in the direction of the normal:
    #   normal_offset > 0 → pen target pushed INTO board (more force)
    #   normal_offset < 0 → pen target pulled out (less force)
    return p_surface - normal_offset * n


def contact_moment_arm(contact_xz: np.ndarray) -> float:
    """
    Distance from board hinge to the contact point (moment arm for torque calc).

    Parameters
    ----------
    contact_xz : ndarray, shape (2,)   Contact position in world (x, z).

    Returns
    -------
    d : float   Moment arm [m].
    """
    return float(np.linalg.norm(contact_xz - BOARD_HINGE_POS))


# ─────────────────────────────────────────────────────────────────────────────
# Inverse Kinematics (Damped Least-Squares + Null-Space)
# ─────────────────────────────────────────────────────────────────────────────

def dls_ik_step(
    theta:           np.ndarray,
    dp_desired:      np.ndarray,
    lambda_damp:     float = 0.02,
    theta_preferred  = None,
    null_gain:       float = 0.3,
    dt:              float = 0.002,
) -> np.ndarray:
    """
    One resolved-rate IK step using Damped Least-Squares (DLS).

    Theory
    ------
    Task-space control law:
        Δθ = J† · Δp_des   (pseudoinverse IK)

    DLS regularisation (avoids singularities):
        J†_dls = Jᵀ (J Jᵀ + λ²·I)⁻¹

    Null-space bias (exploit 3rd DOF):
        Δθ_total = J†_dls · Δp_des + (I − J†_dls · J) · Δθ_null
    where Δθ_null is the secondary-task gradient (e.g. move toward home config).

    Parameters
    ----------
    theta           : current joint angles (3,)
    dp_desired      : desired Δp in world frame (2,)
    lambda_damp     : DLS damping factor
    theta_preferred : preferred (home) joint configuration for null-space
    null_gain       : null-space secondary task gain
    dt              : timestep (used to scale null-space term)

    Returns
    -------
    delta_theta : ndarray, shape (3,)
    """
    J       = jacobian(theta)
    JJT     = J @ J.T
    reg     = JJT + (lambda_damp ** 2) * np.eye(2)

    # Primary task: track dp_desired
    J_dls_pinv = J.T @ np.linalg.solve(reg, np.eye(2))   # J†_dls  (3×2)
    dtheta_primary = J_dls_pinv @ dp_desired

    # Null-space: bias toward preferred configuration (secondary task)
    if theta_preferred is not None:
        N              = np.eye(3) - J_dls_pinv @ J        # null-space projector
        dtheta_null    = null_gain * (theta_preferred - theta)
        dtheta_secondary = N @ dtheta_null
    else:
        dtheta_secondary = np.zeros(3)

    return dtheta_primary + dtheta_secondary


def clamp_joints(theta: np.ndarray) -> np.ndarray:
    """Clamp joint angles to their limits."""
    return np.clip(theta, J_LIMITS[:, 0], J_LIMITS[:, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Analytic IK initialiser (closed-form for 2-link sub-chain)
# ─────────────────────────────────────────────────────────────────────────────

def ik_init_guess(target_xz: np.ndarray, theta3_fixed: float = 0.0) -> np.ndarray:
    """
    Closed-form 2-link IK for joints 1 and 2, given a fixed θ₃.
    Useful to seed the iterative IK solver.

    Uses MuJoCo convention: pz = -sin(α), so wrist offset = L3*[cos, -sin].
    """
    alpha3 = theta3_fixed
    # MuJoCo convention: z = -sin(α)
    p_wrist = target_xz - L3 * np.array([np.cos(alpha3), -np.sin(alpha3)])
    r       = np.linalg.norm(p_wrist)
    r       = np.clip(r, abs(L1 - L2) + 1e-6, L1 + L2 - 1e-6)

    # Law of cosines for elbow angle
    cos_t2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_t2 = np.clip(cos_t2, -1.0, 1.0)
    theta2  = np.arccos(cos_t2)

    # Shoulder angle — atan2(pz, px) but pz = -sin so use p_wrist[1] directly
    psi     = np.arctan2(-p_wrist[1], p_wrist[0])   # negate z back to math frame for atan2
    beta    = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    theta1  = psi - beta

    return clamp_joints(np.array([theta1, theta2, theta3_fixed]))

