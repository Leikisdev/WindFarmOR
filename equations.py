import jax.numpy as jnp
import jax
from dataclasses import dataclass

@jax.tree_util.register_pytree_node_class
@dataclass
class Coordinates:
    x: jnp.ndarray
    y: jnp.ndarray

    def tree_flatten(self):
        children = (self.x, self.y)  # arrays to trace
        aux_data = None              # no static data
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

def BastankhahWakeModel(dx,dy,C_T, k_y, D, mu):
    """
    Calculates the wake velocity deficit caused by another turbine in the farm
    Args:
        dx: Distance between the turbines in the x-direction, along the wind direction
        dy: Distance between the turbines in the y-direction, perpendicular to the wind direction
        C_T: Thrust coefficient
        k_y: Turbulence-dependent scalar
        D: Turbine diameter
    Returns:
        Wake velocity deficit
    """

    sigma = SigmaWakeDef(dx, k_y, D)

    return (1 - jnp.sqrt(1.0 -  C_T / (8.0 * (sigma / D)**2))) * jnp.exp(-0.5 * (dy / (mu * sigma))**2)

def SigmaWakeDef(dx, k_y, D) -> float:
    """
    Calculated the standard deviation of the wake velocity deficit
    """
    return k_y * dx + D / jnp.sqrt(8.0)


def WindFrame(turb_coords: Coordinates, wind_dir_deg: float) -> Coordinates:
    """Convert map coordinates to downwind/crosswind coordinates."""
    wind_dir_deg = 270.0 - wind_dir_deg
    wind_dir_rad = jnp.deg2rad(wind_dir_deg)

    cos_dir = jnp.cos(-wind_dir_rad)
    sin_dir = jnp.sin(-wind_dir_rad)

    x_new = turb_coords.x * cos_dir - turb_coords.y * sin_dir
    y_new = turb_coords.x * sin_dir + turb_coords.y * cos_dir

    return Coordinates(x_new, y_new)

def GaussianWake(frame_coords: Coordinates, turb_diam: float, mu: float) -> jnp.ndarray:
    """Return each turbine's total loss due to wake from upstream turbines."""
    num_turb = frame_coords.x.shape[0]
    CT = 4.0 / 3.0 * (1.0 - 1.0 / 3.0)
    k = 0.0324555

    def loss_from_j(i):
        xi, yi = frame_coords.x[i], frame_coords.y[i]

        def loss_from_target(j):
            x = xi - frame_coords.x[j]
            y = yi - frame_coords.y[j]
            def downstream_case(_):
                return BastankhahWakeModel(x, y, CT, k, turb_diam, mu)
            return jax.lax.cond(x > 0.0, downstream_case, lambda _: 0.0, operand=None)

        loss_array = jax.vmap(loss_from_target)(jnp.arange(num_turb))
        return jnp.sqrt(jnp.sum(loss_array**2))

    return jax.vmap(loss_from_j)(jnp.arange(num_turb))


def DirPower(
    turb_coords: Coordinates,
    wind_dir_deg: float,
    wind_speed: float,
    turb_diam: float,
    turb_ci: float,
    turb_co: float,
    rated_ws: float,
    rated_pwr: float,
    mu: float
) -> jnp.ndarray:
    """Return total power produced by all turbines for one direction."""
    frame_coords = WindFrame(turb_coords, wind_dir_deg)
    loss = GaussianWake(frame_coords, turb_diam, mu)
    wind_speed_eff = wind_speed * (1.0 - loss)

    def power_for_turb(v):
        between_ci_rated = (turb_ci <= v) & (v < rated_ws)
        between_rated_co = (rated_ws <= v) & (v < turb_co)
        pwr_curve = rated_pwr * ((v - turb_ci) / (rated_ws - turb_ci)) ** 3
        pwr = jax.lax.select(between_ci_rated, pwr_curve, 0.0)
        pwr = jax.lax.select(between_rated_co, rated_pwr, pwr)
        return pwr

    turb_pwr = jax.vmap(power_for_turb)(wind_speed_eff)
    return jnp.sum(turb_pwr)


def calcAEP(
    turb_coords: Coordinates,
    wind_freq: jnp.ndarray,
    wind_speed: float,
    wind_dir: jnp.ndarray,
    turb_diam: float,
    turb_ci: float,
    turb_co: float,
    rated_ws: float,
    rated_pwr: float,
    mu: float
) -> jnp.ndarray:
    """Calculate the wind farm AEP for all wind directions."""
    dir_power = jax.vmap(
        lambda wd: DirPower(
            turb_coords, wd, wind_speed,
            turb_diam, turb_ci, turb_co, rated_ws, rated_pwr, mu
        )
    )(wind_dir)

    hrs_per_year = 365.0 * 24.0
    AEP = hrs_per_year * wind_freq * dir_power / 1.0e6  # MWh
    return AEP