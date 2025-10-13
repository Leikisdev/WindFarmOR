import yaml
import jax.numpy as jnp
from equations import Coordinates

def getWindRoseYAML(file_name):
    """Retrieve wind rose data (bins, freqs, speeds) from <.yaml> file."""
    # Read in the .yaml file
    with open(file_name, 'r') as f:
        props = yaml.safe_load(f)['definitions']['wind_inflow']['properties']

    # Fetch wind parameters
    wind_dir = jnp.asarray(props['direction']['bins'])
    wind_freq = jnp.asarray(props['probability']['default'])
    wind_speed = float(props['speed']['default'])

    return wind_dir, wind_freq, wind_speed


def getTurbAtrbtYAML(file_name):
    """Retreive turbine attributes from the <.yaml> file"""
    # Read in the .yaml file
    with open(file_name, 'r') as f:
        defs = yaml.safe_load(f)['definitions']
        op_props = defs['operating_mode']['properties']
        turb_props = defs['wind_turbine_lookup']['properties']
        rotor_props = defs['rotor']['properties']

    # Fetch the turbine parameters
    turb_ci = float(op_props['cut_in_wind_speed']['default'])
    turb_co = float(op_props['cut_out_wind_speed']['default'])
    rated_ws = float(op_props['rated_wind_speed']['default'])
    rated_pwr = float(turb_props['power']['maximum'])
    turb_diam = float(rotor_props['radius']['default']) * 2.

    return turb_ci, turb_co, rated_ws, rated_pwr, turb_diam

def getLocationsYAML(file_name):
    """Retrieve wind turbine locations from <.yaml> file"""
    # Read in the .yaml file
    with open(file_name, 'r') as f:
        positions = yaml.safe_load(f)['definitions']['position']['items']

    # Fetch the x and y coordinates
    x_coords = jnp.asarray(positions['xc'])
    y_coords = jnp.asarray(positions['yc'])
    return Coordinates(x_coords, y_coords)