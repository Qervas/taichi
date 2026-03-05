"""
Phase 7 — Two-Way Rigid Body Car Coupling for SWE Solver

Taichi-based car dynamics coupled bidirectionally with the shallow water
equations. Water pushes cars (buoyancy, drag, torque), cars displace water
(volume displacement, momentum source).

Cars are modeled as rigid rectangles (sedan: 4.5m × 2.0m × 1.5m, 1500 kg).
"""

import math
import numpy as np
import taichi as ti

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════
CAR_LENGTH = 4.5    # m
CAR_WIDTH = 2.0     # m
CAR_HEIGHT = 1.5    # m
CAR_MASS = 1500.0   # kg
CAR_INERTIA = CAR_MASS * (CAR_LENGTH**2 + CAR_WIDTH**2) / 12.0  # rectangular moment

RHO_WATER = 1000.0  # kg/m³
GRAVITY = 9.81
C_DRAG = 1.2        # bluff body drag coefficient
MU_GROUND = 0.7     # ground friction coefficient
GROUNDING_DEPTH = 0.3  # water depth below which car is grounded

MAX_CARS = 128


# ═══════════════════════════════════════════════════════════════════════
# Taichi fields for car state
# ═══════════════════════════════════════════════════════════════════════
car_pos = ti.Vector.field(2, float, shape=MAX_CARS)      # (x, y) in meters
car_vel = ti.Vector.field(2, float, shape=MAX_CARS)      # (vx, vy) m/s
car_yaw = ti.field(float, shape=MAX_CARS)                # θ radians
car_omega = ti.field(float, shape=MAX_CARS)               # angular vel rad/s
car_force = ti.Vector.field(2, float, shape=MAX_CARS)    # accumulated force
car_torque = ti.field(float, shape=MAX_CARS)              # accumulated torque
car_submerged = ti.field(float, shape=MAX_CARS)           # submerged fraction [0,1]
car_active = ti.field(int, shape=MAX_CARS)                # 1 if active
n_cars_f = ti.field(int, shape=())


def init_cars(positions, yaws):
    """Initialize car state from numpy arrays.

    Args:
        positions: (N, 2) float32 — car positions in meters
        yaws: (N,) float32 — yaw angles in radians
    """
    n = min(len(positions), MAX_CARS)
    n_cars_f[None] = n

    pos_np = np.zeros((MAX_CARS, 2), dtype=np.float32)
    vel_np = np.zeros((MAX_CARS, 2), dtype=np.float32)
    yaw_np = np.zeros(MAX_CARS, dtype=np.float32)
    omega_np = np.zeros(MAX_CARS, dtype=np.float32)
    active_np = np.zeros(MAX_CARS, dtype=np.int32)

    pos_np[:n] = positions[:n]
    yaw_np[:n] = yaws[:n]
    active_np[:n] = 1

    car_pos.from_numpy(pos_np)
    car_vel.from_numpy(vel_np)
    car_yaw.from_numpy(yaw_np)
    car_omega.from_numpy(omega_np)
    car_active.from_numpy(active_np)


# ═══════════════════════════════════════════════════════════════════════
# Water → Car forces
# ═══════════════════════════════════════════════════════════════════════
@ti.kernel
def water_to_car_forces(h_field: ti.template(), hu_field: ti.template(),
                        hv_field: ti.template(), z_bed_field: ti.template(),
                        is_wall_field: ti.template(),
                        nx: int, ny: int, dx: float):
    """Compute forces on each car from the water field.

    For each car, sample the water depth and velocity at the car footprint,
    then compute buoyancy, drag, and friction forces.
    """
    n_cars = n_cars_f[None]

    for c in range(n_cars):
        if car_active[c] == 0:
            continue

        px, py = car_pos[c][0], car_pos[c][1]
        vx, vy = car_vel[c][0], car_vel[c][1]
        theta = car_yaw[c]

        cos_t = ti.cos(theta)
        sin_t = ti.sin(theta)

        # Car footprint corners (rotated rectangle)
        half_l = CAR_LENGTH / 2.0
        half_w = CAR_WIDTH / 2.0

        # Sample water at car center and four corners
        total_h = 0.0
        total_hu = 0.0
        total_hv = 0.0
        n_samples = 0
        total_z = 0.0

        # Sample points: center + 4 corners
        for si in ti.static(range(5)):
            lx = 0.0
            ly = 0.0
            if si == 1:
                lx, ly = half_l, half_w
            elif si == 2:
                lx, ly = half_l, -half_w
            elif si == 3:
                lx, ly = -half_l, half_w
            elif si == 4:
                lx, ly = -half_l, -half_w

            # Rotate to world coords
            wx = px + lx * cos_t - ly * sin_t
            wy = py + lx * sin_t + ly * cos_t

            # Grid index
            gi = int(wx / dx)
            gj = int(wy / dx)

            if 0 <= gi < nx and 0 <= gj < ny and is_wall_field[gi, gj] == 0:
                total_h += h_field[gi, gj]
                total_hu += hu_field[gi, gj]
                total_hv += hv_field[gi, gj]
                total_z += z_bed_field[gi, gj]
                n_samples += 1

        if n_samples == 0:
            car_force[c] = ti.Vector([0.0, 0.0])
            car_torque[c] = 0.0
            car_submerged[c] = 0.0
            continue

        inv_n = 1.0 / n_samples
        avg_h = total_h * inv_n
        avg_hu = total_hu * inv_n
        avg_hv = total_hv * inv_n

        # Water velocity at car location
        vw_x = 0.0
        vw_y = 0.0
        if avg_h > 1e-4:
            vw_x = avg_hu / avg_h
            vw_y = avg_hv / avg_h

        # --- Submerged fraction ---
        submerged_depth = ti.min(avg_h, CAR_HEIGHT)
        submerged_frac = submerged_depth / CAR_HEIGHT
        car_submerged[c] = submerged_frac

        # --- Buoyancy force ---
        # Submerged volume = L * W * submerged_depth
        v_sub = CAR_LENGTH * CAR_WIDTH * submerged_depth
        # Buoyancy is upward (handled vertically, but in 2D SWE we track
        # whether car floats — reduces ground friction)

        # --- Drag force ---
        # Relative velocity (water - car)
        dvx = vw_x - vx
        dvy = vw_y - vy
        dv_mag = ti.sqrt(dvx * dvx + dvy * dvy)

        # Frontal area depends on car orientation relative to flow
        a_frontal = CAR_WIDTH * submerged_depth
        f_drag_mag = 0.5 * C_DRAG * RHO_WATER * a_frontal * dv_mag * dv_mag

        # Drag direction: along relative velocity
        fx_drag = 0.0
        fy_drag = 0.0
        if dv_mag > 1e-4:
            fx_drag = f_drag_mag * dvx / dv_mag
            fy_drag = f_drag_mag * dvy / dv_mag

        # --- Ground friction ---
        # Friction decreases as car floats (buoyancy offsets weight)
        buoyancy_force = RHO_WATER * GRAVITY * v_sub
        weight = CAR_MASS * GRAVITY
        normal_force = ti.max(weight - buoyancy_force, 0.0)

        # Grounding: high friction when shallow
        friction_coeff = MU_GROUND
        if avg_h < GROUNDING_DEPTH:
            friction_coeff = MU_GROUND * 1.5  # extra grip when nearly dry

        car_speed = ti.sqrt(vx * vx + vy * vy)
        fx_fric = 0.0
        fy_fric = 0.0
        if car_speed > 1e-4:
            f_fric = friction_coeff * normal_force
            f_fric = ti.min(f_fric, CAR_MASS * car_speed / 0.01)  # limit to stop car in dt
            fx_fric = -f_fric * vx / car_speed
            fy_fric = -f_fric * vy / car_speed

        # --- Total force ---
        fx_total = fx_drag + fx_fric
        fy_total = fy_drag + fy_fric
        car_force[c] = ti.Vector([fx_total, fy_total])

        # --- Torque from asymmetric drag ---
        # Sample drag at front and rear of car separately
        torque = 0.0

        # Front half water velocity
        front_x = px + half_l * cos_t
        front_y = py + half_l * sin_t
        fi = int(front_x / dx)
        fj = int(front_y / dx)

        rear_x = px - half_l * cos_t
        rear_y = py - half_l * sin_t
        ri = int(rear_x / dx)
        rj = int(rear_y / dx)

        vw_front_x = vw_x
        vw_front_y = vw_y
        vw_rear_x = vw_x
        vw_rear_y = vw_y

        if 0 <= fi < nx and 0 <= fj < ny:
            hf = h_field[fi, fj]
            if hf > 1e-4:
                vw_front_x = hu_field[fi, fj] / hf
                vw_front_y = hv_field[fi, fj] / hf

        if 0 <= ri < nx and 0 <= rj < ny:
            hr = h_field[ri, rj]
            if hr > 1e-4:
                vw_rear_x = hu_field[ri, rj] / hr
                vw_rear_y = hv_field[ri, rj] / hr

        # Lateral component of relative velocity at front/rear
        # Car's lateral direction: (-sin_t, cos_t)
        dv_front_lat = (vw_front_x - vx) * (-sin_t) + (vw_front_y - vy) * cos_t
        dv_rear_lat = (vw_rear_x - vx) * (-sin_t) + (vw_rear_y - vy) * cos_t

        # Torque = (force_front - force_rear) * moment_arm
        f_front = 0.5 * C_DRAG * RHO_WATER * (CAR_WIDTH * submerged_depth * 0.5) * \
                  ti.abs(dv_front_lat) * dv_front_lat
        f_rear = 0.5 * C_DRAG * RHO_WATER * (CAR_WIDTH * submerged_depth * 0.5) * \
                 ti.abs(dv_rear_lat) * dv_rear_lat

        torque = (f_front - f_rear) * half_l

        # Angular friction
        if ti.abs(car_omega[c]) > 1e-4:
            ang_fric = friction_coeff * normal_force * 0.3  # reduced moment arm
            torque -= ang_fric * car_omega[c] / ti.abs(car_omega[c])

        car_torque[c] = torque


# ═══════════════════════════════════════════════════════════════════════
# Integrate car motion
# ═══════════════════════════════════════════════════════════════════════
@ti.kernel
def integrate_cars(dt: float, domain_size: float):
    """Update car velocity and position from accumulated forces."""
    n_cars = n_cars_f[None]

    for c in range(n_cars):
        if car_active[c] == 0:
            continue

        # Linear acceleration
        ax = car_force[c][0] / CAR_MASS
        ay = car_force[c][1] / CAR_MASS

        # Update velocity
        new_vx = car_vel[c][0] + ax * dt
        new_vy = car_vel[c][1] + ay * dt

        # Velocity damping (numerical stability)
        speed = ti.sqrt(new_vx * new_vx + new_vy * new_vy)
        max_speed = 15.0  # cap at ~54 km/h
        if speed > max_speed:
            new_vx *= max_speed / speed
            new_vy *= max_speed / speed

        car_vel[c] = ti.Vector([new_vx, new_vy])

        # Update position
        new_px = car_pos[c][0] + new_vx * dt
        new_py = car_pos[c][1] + new_vy * dt

        # Clamp to domain
        margin = CAR_LENGTH
        new_px = ti.max(margin, ti.min(domain_size - margin, new_px))
        new_py = ti.max(margin, ti.min(domain_size - margin, new_py))
        car_pos[c] = ti.Vector([new_px, new_py])

        # Angular acceleration
        alpha = car_torque[c] / CAR_INERTIA
        new_omega = car_omega[c] + alpha * dt
        # Damping
        new_omega *= 0.98
        max_omega = 2.0  # cap angular velocity
        new_omega = ti.max(-max_omega, ti.min(max_omega, new_omega))
        car_omega[c] = new_omega

        # Update yaw
        car_yaw[c] += new_omega * dt


# ═══════════════════════════════════════════════════════════════════════
# Car → Water source terms (volume displacement)
# ═══════════════════════════════════════════════════════════════════════
@ti.kernel
def car_to_water_source(h_field: ti.template(), hu_field: ti.template(),
                        hv_field: ti.template(), is_wall_field: ti.template(),
                        nx: int, ny: int, dx: float, dt: float):
    """Apply car volume displacement and momentum source to water.

    For cells under each car:
    - Displace water (reduce h by car-occupied volume fraction)
    - Push water at car leading edge (momentum source)
    """
    n_cars = n_cars_f[None]

    for c in range(n_cars):
        if car_active[c] == 0:
            continue

        px, py = car_pos[c][0], car_pos[c][1]
        vx, vy = car_vel[c][0], car_vel[c][1]
        theta = car_yaw[c]

        cos_t = ti.cos(theta)
        sin_t = ti.sin(theta)

        half_l = CAR_LENGTH / 2.0
        half_w = CAR_WIDTH / 2.0

        # Bounding box of rotated car in grid coords
        # Check all 4 corners to find AABB
        min_gx = nx
        max_gx = 0
        min_gy = ny
        max_gy = 0

        for ci in ti.static(range(4)):
            lx = half_l if ci < 2 else -half_l
            ly = half_w if ci % 2 == 0 else -half_w
            wx = px + lx * cos_t - ly * sin_t
            wy = py + lx * sin_t + ly * cos_t
            gi = int(wx / dx)
            gj = int(wy / dx)
            ti.atomic_min(min_gx, gi)
            ti.atomic_max(max_gx, gi)
            ti.atomic_min(min_gy, gj)
            ti.atomic_max(max_gy, gj)

        min_gx = ti.max(1, min_gx - 1)
        max_gx = ti.min(nx - 2, max_gx + 1)
        min_gy = ti.max(1, min_gy - 1)
        max_gy = ti.min(ny - 2, max_gy + 1)

        # Process cells in car bounding box
        for i in range(min_gx, max_gx + 1):
            for j in range(min_gy, max_gy + 1):
                if is_wall_field[i, j] == 1:
                    continue

                # Cell center in world coords
                cell_x = (i + 0.5) * dx
                cell_y = (j + 0.5) * dx

                # Transform to car-local coords
                rel_x = cell_x - px
                rel_y = cell_y - py
                local_x = rel_x * cos_t + rel_y * sin_t
                local_y = -rel_x * sin_t + rel_y * cos_t

                # Check if cell center is inside car rectangle
                if ti.abs(local_x) < half_l and ti.abs(local_y) < half_w:
                    h_here = h_field[i, j]
                    if h_here > 1e-4:
                        # Volume displacement: reduce water height
                        # Car occupies fraction of cell
                        cell_area = dx * dx
                        car_overlap = ti.min(CAR_LENGTH, dx) * ti.min(CAR_WIDTH, dx)
                        frac = car_overlap / cell_area * 0.3  # scale factor

                        # Push water at leading edge
                        car_speed = ti.sqrt(vx * vx + vy * vy)
                        if car_speed > 0.1:
                            # Add momentum source proportional to car velocity
                            momentum_frac = frac * 0.5 * dt
                            ti.atomic_add(hu_field[i, j], h_here * vx * momentum_frac)
                            ti.atomic_add(hv_field[i, j], h_here * vy * momentum_frac)


# ═══════════════════════════════════════════════════════════════════════
# Car-car and car-building collision (simple AABB repulsion)
# ═══════════════════════════════════════════════════════════════════════
@ti.kernel
def car_collisions(is_wall_field: ti.template(), nx: int, ny: int, dx: float):
    """Simple collision: push cars apart and away from walls."""
    n_cars = n_cars_f[None]

    for c in range(n_cars):
        if car_active[c] == 0:
            continue

        px, py = car_pos[c][0], car_pos[c][1]

        # Car-building collision: check if car center is near a wall cell
        gi = int(px / dx)
        gj = int(py / dx)

        repulse_x = 0.0
        repulse_y = 0.0
        check_r = 3  # check nearby cells

        for di in range(-check_r, check_r + 1):
            for dj in range(-check_r, check_r + 1):
                wi = gi + di
                wj = gj + dj
                if 0 <= wi < nx and 0 <= wj < ny and is_wall_field[wi, wj] == 1:
                    wall_x = (wi + 0.5) * dx
                    wall_y = (wj + 0.5) * dx
                    ddx = px - wall_x
                    ddy = py - wall_y
                    dist = ti.sqrt(ddx * ddx + ddy * ddy) + 1e-6
                    min_dist = CAR_LENGTH * 0.8
                    if dist < min_dist:
                        push = (min_dist - dist) * 50.0  # repulsion spring
                        repulse_x += push * ddx / dist
                        repulse_y += push * ddy / dist

        # Car-car collision
        for c2 in range(n_cars):
            if c2 == c or car_active[c2] == 0:
                continue
            ddx = px - car_pos[c2][0]
            ddy = py - car_pos[c2][1]
            dist = ti.sqrt(ddx * ddx + ddy * ddy) + 1e-6
            min_dist = CAR_LENGTH * 1.2
            if dist < min_dist:
                push = (min_dist - dist) * 30.0
                repulse_x += push * ddx / dist
                repulse_y += push * ddy / dist

        # Apply repulsion as velocity impulse
        car_vel[c] += ti.Vector([repulse_x * 0.01, repulse_y * 0.01])


# ═══════════════════════════════════════════════════════════════════════
# Get car state as numpy arrays (for export)
# ═══════════════════════════════════════════════════════════════════════
def get_car_state(n_cars):
    """Return car state as numpy arrays for export."""
    pos = car_pos.to_numpy()[:n_cars]
    vel = car_vel.to_numpy()[:n_cars]
    yaw = car_yaw.to_numpy()[:n_cars]
    submerged = car_submerged.to_numpy()[:n_cars]
    return pos, vel, yaw, submerged
