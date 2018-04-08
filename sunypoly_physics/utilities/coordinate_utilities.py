import numpy as np

def spherical_to_cartesian(radial, azimuthal, polar):
    x = radial * np.cos(azimuthal) * np.sin(polar)
    y = radial * np.sin(azimuthal) * np.sin(polar)
    z = radial * np.cos(polar)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    radial = np.sqrt(x**2 + y**2 + z**2)
    azimuthal = np.arctan2(x1=y, x2=x)
    polar = np.arccos(z / radial)
    return radial, azimuthal, polar