"""
controller.py — Chopstick Crane Controller
===========================================
Implements the closed-loop control stack for the 3-link arm:

Control Layers
--------------
1. **PD Joint Control**      (hardware level, in MuJoCo actuators)
2. **Task-Space / DLS IK**   (kinematic control loop, this file)
3. **Admittance Force Ctrl** (force regulation via position offset)
4. **Null-Space Control**    (redundancy — keep arm near home config)

Phase Sequence
--------------
  WARMUP   -> static hold (board settles, contact established)
  SWEEP    -> sinusoidal traversal s: 0->1 (one full cycle)
  RETURN   -> smooth return to home joint configuration

Usage
-----
    ctrl = ChopstickController(model, data)
    while sim_running:
        ctrl.step(data)        # updates data.ctrl with new joint targets
        mujoco.mj_step(model, data)
"""

import numpy as np
import mujoco
from enum import Enum, auto

from kinematics import (
    forward_kinematics, jacobian, dls_ik_step, clamp_joints,
    target_with_offset, ik_init_guess,
    BOARD_HINGE_POS, F_MIN, F_MAX, F_TARGET,
    SWEEP_AMPLITUDE, SWEEP_OFFSET,
)


# ─────────────────────────────────────────────────────────────────────────────
# Phase definitions
# ─────────────────────────────────────────────────────────────────────────────

class Phase(Enum):
    WARMUP  = auto()   # Static hold: pen settles onto board
    SWEEP   = auto()   # Sinusoidal sweep along board surface
    RETURN  = auto()   # Return to home configuration
    DONE    = auto()


PHASE_DURATION = {
    Phase.WARMUP : 2.0,   # seconds
    Phase.SWEEP  : 35.0,  # Continuous back-and-forth for 35 seconds
    Phase.RETURN : 1.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class ChopstickController:
    """
    Full controller for the Chopstick Crane simulation.

    Parameters
    ----------
    model : mujoco.MjModel
    data  : mujoco.MjData
    dt    : float   Simulation timestep [s].
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, dt: float = 0.002):
        self.model = model
        self.data  = data
        self.dt    = dt

        # ── Sensor addresses (sensordata offsets, not IDs) ──
        # Must use model.sensor_adr[sensor_id] to get the correct offset
        # into data.sensordata[] for each sensor.
        self._adr_j1       = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "s_j1")]
        self._adr_j2       = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "s_j2")]
        self._adr_j3       = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "s_j3")]
        self._adr_phi      = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "s_phi")]
        self._adr_phi_dot  = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "s_phi_dot")]
        self._adr_tip      = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "s_tip_pos")]
        self._adr_touch    = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "s_touch")]

        # ── Actuator ids ──
        self._act_j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_j1")
        self._act_j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_j2")
        self._act_j3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_j3")

        # ── Geom ids for contact force reading ──
        self._pen_geom   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pen_sphere")
        self._board_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "board_surface")

        # ── Mocap id for visual marker ──
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_mocap")
        self._target_mocap_id = model.body_mocapid[body_id] if body_id >= 0 else -1

        # ── State ──
        self.phase         = Phase.WARMUP
        self.phase_time    = 0.0         # time within current phase
        self.sim_time      = 0.0

        # Home config: arm reaching toward board, elbow down
        self._theta_home   = self._compute_home_config()
        self._theta_cmd    = self._theta_home.copy()  # commanded joints

        # Sweep state
        self._s            = 0.0         # sweep parameter s ∈ [0, 1]
        self._normal_off   = 0.004       # force-regulation offset [m] (into board)

        # Admittance control gains
        self._kf           = 0.0015      # force -> position gain [m/N]
        self._kf_i         = 0.0         # (integral disabled to prevent windup)
        self._force_int    = 0.0         # force error integral

        # DLS IK parameters
        self._lambda       = 0.010       # DLS damping (reduced for slightly faster tracking)
        self._null_gain    = 0.2         # null-space secondary task gain

        # Logging (filled during simulation)
        self.log_t         = []
        self.log_theta     = []
        self.log_tip       = []
        self.log_phi       = []
        self.log_fn        = []
        self.log_target    = []
        self.log_phase     = []
        self.log_s         = []

        # Initialise joint commands to home
        self._set_actuators(self._theta_home)
        print(f"[Controller] Home config: theta = {np.degrees(self._theta_home).round(1)} deg")

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance controller by one timestep. Call BEFORE mujoco.mj_step()."""
        # 1. Read state from sensors
        theta, phi, phi_dot, tip_xz, touch = self._read_state()

        # 2. Estimate contact normal force
        Fn = self._estimate_contact_force(phi, phi_dot)

        # 3. Run phase logic -> compute new joint targets
        self._run_phase(theta, phi, Fn)

        # 4. Write actuator commands
        self._set_actuators(self._theta_cmd)

        # 5. Log everything
        self._log(theta, phi, tip_xz, Fn)

        self.phase_time += self.dt
        self.sim_time   += self.dt

    @property
    def done(self) -> bool:
        return self.phase == Phase.DONE

    def get_logs(self) -> dict:
        return {
            "t"      : np.array(self.log_t),
            "theta"  : np.array(self.log_theta),
            "tip"    : np.array(self.log_tip),
            "phi"    : np.array(self.log_phi),
            "Fn"     : np.array(self.log_fn),
            "target" : np.array(self.log_target),
            "phase"  : np.array(self.log_phase),
            "s"      : np.array(self.log_s),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Phase logic
    # ─────────────────────────────────────────────────────────────────────────

    def _run_phase(self, theta: np.ndarray, phi: float, Fn: float) -> None:
        """Dispatch to phase-specific controllers."""

        if self.phase == Phase.WARMUP:
            self._phase_warmup(theta, phi, Fn)
            if self.phase_time >= PHASE_DURATION[Phase.WARMUP]:
                print(f"[t={self.sim_time:.2f}s] -> SWEEP phase")
                self.phase      = Phase.SWEEP
                self.phase_time = 0.0
                self._s         = 0.0

        elif self.phase == Phase.SWEEP:
            self._phase_sweep(theta, phi, Fn)
            if self.phase_time >= PHASE_DURATION[Phase.SWEEP]:
                print(f"[t={self.sim_time:.2f}s] -> RETURN phase")
                self.phase      = Phase.RETURN
                self.phase_time = 0.0

        elif self.phase == Phase.RETURN:
            self._phase_return(theta)
            if self.phase_time >= PHASE_DURATION[Phase.RETURN]:
                print(f"[t={self.sim_time:.2f}s] -> DONE")
                self.phase = Phase.DONE

    # ── Phase: WARMUP ──

    def _phase_warmup(self, theta: np.ndarray, phi: float, Fn: float) -> None:
        """
        Move to s=0 position, make gentle contact.
        Admittance only activates once contact is detected (Fn > 0.1N),
        preventing pre-contact offset windup.
        """
        s = 0.0
        if Fn > 0.1:   # only run admittance when actually in contact
            self._normal_off = self._admittance_update(Fn, self._normal_off)
        # else: hold at current offset (initially 0.004 m — just enough to make contact)
        p_target = target_with_offset(s, phi, self._normal_off)
        self._ik_update(theta, p_target)

    # ── Phase: SWEEP ──

    def _phase_sweep(self, theta: np.ndarray, phi: float, Fn: float) -> None:
        """
        Sinusoidal sweep: s sweeps continuously back and forth.
        At each step:
          (a) Update s (sweep parameter)
          (b) Update normal offset via admittance control (force regulation)
          (c) Compute world-frame target from board frame (with current phi)
          (d) Run DLS IK step
        """
        import math
        # Sweep back and forth every 10 seconds (5s forward, 5s backward)
        self._s = 0.5 - 0.5 * math.cos(math.pi * self.phase_time / 5.0)

        # Admittance control: adjust penetration to regulate force
        self._normal_off = self._admittance_update(Fn, self._normal_off)

        # Target: on the sinusoidal curve, compensating for board tilt
        p_target = target_with_offset(self._s, phi, self._normal_off)
        self._ik_update(theta, p_target)

    # ── Phase: RETURN ──

    def _phase_return(self, theta: np.ndarray) -> None:
        """
        Smooth interpolation from current commanded config to home config.
        Uses joint-space interpolation (no force control needed here).
        """
        alpha           = min(self.phase_time / PHASE_DURATION[Phase.RETURN], 1.0)
        theta_target    = (1 - alpha) * self._theta_cmd + alpha * self._theta_home
        self._theta_cmd = clamp_joints(theta_target)

    # ─────────────────────────────────────────────────────────────────────────
    # IK update (DLS + null-space)
    # ─────────────────────────────────────────────────────────────────────────

    def _ik_update(self, theta: np.ndarray, p_target: np.ndarray) -> None:
        """
        Compute dtheta via DLS IK and update self._theta_cmd.

        Resolved-rate control law:
            dp   = p_target − FK(theta_cmd)   -> use COMMANDED theta, not sensor theta
            dtheta   = J†_dls · dp + (I − J†_dls·J) · dtheta_null
            theta_cmd_new = theta_cmd + dtheta   (clamped to limits)

        IMPORTANT: We integrate from self._theta_cmd (the last commanded position)
        rather than the sensor-read theta. This prevents the lag feedback loop
        where the controller always chases from behind due to PD tracking delay.
        The sensor theta is used only for the Jacobian (current arm geometry).
        """
        # Use commanded theta for FK (where we WANT to be)
        p_current = forward_kinematics(self._theta_cmd)
        dp        = p_target - p_current

        # Clip max step size (stability)
        max_step  = 0.05     # [m] per timestep
        dp_norm   = np.linalg.norm(dp)
        if dp_norm > max_step:
            dp = dp * (max_step / dp_norm)

        # Use actual theta for Jacobian (true arm geometry for IK accuracy)
        dtheta = dls_ik_step(
            theta,              # actual joints for correct Jacobian
            dp,
            lambda_damp     = self._lambda,
            theta_preferred = self._theta_home,
            null_gain       = self._null_gain,
            dt              = self.dt,
        )

        self._theta_cmd = clamp_joints(self._theta_cmd + dtheta)

    # ─────────────────────────────────────────────────────────────────────────
    # Admittance Control (Force Regulation)
    # ─────────────────────────────────────────────────────────────────────────

    def _admittance_update(self, Fn: float, normal_off_current: float) -> float:
        """
        Admittance control: adjust normal offset to regulate contact force.

        Law:  dd = Kf · (Fn* − Fn)
              d_new = d_old + dd

        If Fn < F_MIN -> increase penetration (push harder)
        If Fn > F_MAX -> decrease penetration (back off)

        This is an impedance/admittance control paradigm:
        force error -> position command adjustment.
        """
        force_error       = F_TARGET - Fn
        delta_off         = self._kf * force_error
        new_off           = normal_off_current + delta_off
        # Clamp offset to sensible range: [0, 2 cm penetration]
        return float(np.clip(new_off, 0.0, 0.020))

    # ─────────────────────────────────────────────────────────────────────────
    # Contact Force Estimation
    # ─────────────────────────────────────────────────────────────────────────

    def _estimate_contact_force(self, phi: float, phi_dot: float) -> float:
        """
        Estimate contact normal force from two methods and blend.

        Method 1 — Dynamics-based:
            The board is a 2nd-order system:
                I·phï = τ_contact − k·phi − b·phi̇
            Reading from sensordata gives phi and phi_dot.
            We use the board joint's qfrc_constraint from MuJoCo.

        Method 2 — MuJoCo contact data:
            Directly read contact force from data.contact array.

        We try method 2 first (most accurate), fall back to method 1.
        """
        Fn = self._read_contact_force_mujoco()
        if Fn < 1e-6:
            # Fall back: estimate from board spring torque
            # At quasi-static equilibrium: k·phi ≈ Fn · d
            # where d ≈ SWEEP_OFFSET (moment arm approximation)
            d  = SWEEP_OFFSET + SWEEP_AMPLITUDE * np.sin(2 * np.pi * self._s)
            d  = max(d, 0.05)   # avoid division by zero
            Fn = max((self.model.jnt_stiffness[self._board_jnt_id] * abs(phi)) / d, 0.0)
        return float(Fn)

    def _read_contact_force_mujoco(self) -> float:
        """
        Sum normal contact forces between pen_sphere and board_surface.
        """
        total = 0.0
        c_force = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if ( (c.geom[0] == self._pen_geom and c.geom[1] == self._board_geom) or
                 (c.geom[0] == self._board_geom and c.geom[1] == self._pen_geom) ):
                mujoco.mj_contactForce(self.model, self.data, i, c_force)
                total += abs(c_force[0])   # normal component in contact frame
        return total

    # ─────────────────────────────────────────────────────────────────────────
    # State reading
    # ─────────────────────────────────────────────────────────────────────────

    def _read_state(self):
        """Read current state from MuJoCo sensors."""
        sd      = self.data.sensordata
        theta   = np.array([sd[self._adr_j1], sd[self._adr_j2], sd[self._adr_j3]])
        phi     = float(sd[self._adr_phi])
        phi_dot = float(sd[self._adr_phi_dot])
        # tip_pos is [x, y, z]; we only need x (offset+0) and z (offset+2)
        tip_adr = self._adr_tip
        tip_xz  = np.array([sd[tip_adr], sd[tip_adr + 2]])
        touch   = float(sd[self._adr_touch])
        return theta, phi, phi_dot, tip_xz, touch

    # ─────────────────────────────────────────────────────────────────────────
    # Actuator commands
    # ─────────────────────────────────────────────────────────────────────────

    def _set_actuators(self, theta_cmd: np.ndarray) -> None:
        self.data.ctrl[self._act_j1] = theta_cmd[0]
        self.data.ctrl[self._act_j2] = theta_cmd[1]
        self.data.ctrl[self._act_j3] = theta_cmd[2]

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_home_config(self) -> np.ndarray:
        """
        Compute a home joint configuration where the pen is near the
        board's s=0 target position.
        """
        from kinematics import BOARD_HINGE_POS, SWEEP_OFFSET, target_curve
        p_target = target_curve(0.0, 0.0)
        theta0   = ik_init_guess(p_target, theta3_fixed=0.0)
        # Refine with a few IK iterations
        for _ in range(200):
            p_cur  = forward_kinematics(theta0)
            dp     = p_target - p_cur
            if np.linalg.norm(dp) < 1e-5:
                break
            dtheta = dls_ik_step(theta0, dp, lambda_damp=0.02)
            theta0 = clamp_joints(theta0 + dtheta)
        return theta0

    @property
    def _board_jnt_id(self) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "board_hinge")

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log(self, theta, phi, tip_xz, Fn) -> None:
        from kinematics import target_curve
        self.log_t.append(self.sim_time)
        self.log_theta.append(theta.copy())
        self.log_tip.append(tip_xz.copy())
        self.log_phi.append(phi)
        self.log_fn.append(Fn)
        target_pos = target_curve(self._s, phi).copy()
        if self._target_mocap_id >= 0:
            self.data.mocap_pos[self._target_mocap_id][0] = target_pos[0]
            self.data.mocap_pos[self._target_mocap_id][2] = target_pos[1]
        self.log_target.append(target_pos)
        self.log_phase.append(self.phase.value)
        self.log_s.append(self._s)
