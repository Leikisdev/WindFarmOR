import yaml
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from equations import calcAEP
from data_extract import *
import numpy as np

# Input paths
input_path = "input_data/"
windrose_file = "iea37-windrose.yaml"
wind_spec_file = "iea37-335mw.yaml"

# Output paths
output_path = "output_data/"
output_template = "iea37-output-template.yaml"

def runOptimization(input_loc_path, farm_radius, mu = 1.0):
    # Load data
    init_turb_coords = getLocationsYAML(input_loc_path)
    wind_dir, wind_freq, wind_speed = getWindRoseYAML(f"{input_path}{windrose_file}")
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML(f"{input_path}{wind_spec_file}")

    # 
    n_turb = init_turb_coords.x.size
    xy0 = np.asarray(flatten_coords(init_turb_coords), dtype=np.float64)
    if xy0[0] == 0 and xy0[n_turb] == 0:
        xy0[0] = 1e-3
        xy0[n_turb] = 1e-3

    # Define objective in terms of Coordinates
    def objective_func(coords: Coordinates):
        return -jnp.sum(calcAEP(
            turb_coords=coords,
            wind_freq=wind_freq,
            wind_dir=wind_dir,
            wind_speed=wind_speed,
            turb_ci=turb_ci,
            turb_co=turb_co,
            rated_ws=rated_ws,
            rated_pwr=rated_pwr,
            turb_diam=turb_diam,
            mu=mu
        ))

    # Constraint: turbines must stay within radius
    def constraint_perim(coords: Coordinates):
        return farm_radius - jnp.sqrt(coords.x**2 + coords.y**2)
    
    def constraint_prox(coords):
        dx = coords.x[:, None] - coords.x
        dy = coords.y[:, None] - coords.y
        dist = jnp.sqrt(dx**2 + dy**2)
        i_idx, j_idx = jnp.triu_indices(len(coords.x), k=1)
        return dist[i_idx, j_idx] - turb_diam * 2
    
    # JAX derivatives
    grad_objective = jax.grad(objective_func)
    jac_constraint_perim = jax.jacobian(constraint_perim)
    jac_constraint_prox = jax.jacobian(constraint_prox)


    def objective_np(xy_flat):
        coords = unflatten_coords(jnp.asarray(xy_flat), n_turb)
        val = objective_func(coords)
        return float(val) # safe Python scalar

    def grad_np(xy_flat):
        coords = unflatten_coords(jnp.asarray(xy_flat), n_turb)
        grad_coords = grad_objective(coords)
        return np.concatenate([
            np.asarray(grad_coords.x, dtype=np.float64),
            np.asarray(grad_coords.y, dtype=np.float64)
        ])

    def constraint_np(xy_flat):
        coords = unflatten_coords(jnp.asarray(xy_flat), n_turb)
        return np.asarray((farm_radius - jnp.sqrt(coords.x**2 + coords.y**2)), dtype=np.float64)
    
    def constraint_prox_np(xy_flat):
        coords = unflatten_coords(jnp.asarray(xy_flat), n_turb)
        vals = constraint_prox(coords)
        return np.asarray(vals, dtype=np.float64)

    def jac_np(xy_flat):
        coords = unflatten_coords(jnp.asarray(xy_flat), n_turb)
        jacobian_coords = jac_constraint_perim(coords)
        return np.concatenate([
            np.asarray(jacobian_coords.x, dtype=np.float64),
            np.asarray(jacobian_coords.y, dtype=np.float64)
        ], axis=1)
    
    def jac_prox_np(xy_flat):
        coords = unflatten_coords(jnp.asarray(xy_flat), n_turb)
        jacobian_coords = jac_constraint_prox(coords)
        return np.concatenate([
            np.asarray(jacobian_coords.x, dtype=np.float64),
            np.asarray(jacobian_coords.y, dtype=np.float64)
        ], axis=1)

    cons = [
        {'type': 'ineq', 'fun': constraint_np, 'jac': jac_np},
        {'type': 'ineq', 'fun': constraint_prox_np, 'jac': jac_prox_np}
    ]

    # Run optimization
    res = minimize(objective_np, xy0, jac=grad_np,
                   constraints=cons, method='SLSQP',options={'ftol': 1e-6, 'disp': True, 'maxiter': 1000})
    if res.success:
        # Return optimized coordinates results as Coordinates object again
        opt_coords = unflatten_coords(res.x, n_turb)

        # Calculate AEP vector for optimized coordinates
        AEP_array = calcAEP(
            opt_coords,
            wind_freq=wind_freq,
            wind_dir=wind_dir,
            wind_speed=wind_speed,
            turb_ci=turb_ci,
            turb_co=turb_co,
            rated_ws=rated_ws,
            rated_pwr=rated_pwr,
            turb_diam=turb_diam,
            mu=1.0
        )

        storeOutput(opt_coords.x, opt_coords.y, AEP_array, jnp.sum(AEP_array), n_turb, mu)
        return opt_coords, res
    else:
        print(res)
    return Coordinates(jnp.empty(0), jnp.empty(0)), res

def flatten_coords(coords: Coordinates) -> jnp.ndarray:
    """Convert Coordinates → flat array."""
    return jnp.concatenate([coords.x, coords.y])

def unflatten_coords(xy_flat: jnp.ndarray, n_turb: int) -> Coordinates:
    """Convert flat array → Coordinates."""
    return Coordinates(
        x = xy_flat[:n_turb],
        y = xy_flat[n_turb:]
    )

def storeOutput(x_coords, y_coords, AEP_array, AEP_total, num_turb, mu):
    # 1. Read the YAML template file
    with open(output_path + output_template, "r") as f:
        data = yaml.safe_load(f)

    # 2. Modify the data
    data['definitions']['position']['items']['xc'] = x_coords.tolist()
    data['definitions']['position']['items']['yc'] = y_coords.tolist()
    data['definitions']['plant_energy']['properties']['annual_energy_production']['default'] = float(AEP_total)
    data['definitions']['plant_energy']['properties']['annual_energy_production']['binned'] = AEP_array.tolist()


    # 3. Save to a new YAML file
    with open(f"{output_path}iea37-{num_turb}-{int(mu * 10)}.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

if __name__ == "__main__":
    coords, res = runOptimization(f"{output_path}iea37-16-18.yaml", 1300, 1.7)
    if res.success:
        print("Found solution:")
        print(f"X coords: {coords.x}")
        print(f"Y coords: {coords.y}")
    else:
        print(f"Error: {res.message}")
    