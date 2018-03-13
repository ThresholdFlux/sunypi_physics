from math import pi, atan  # cos, sqrt, sin, tan

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


if __name__ == '__main__':
    radius_sphere = 1.0
    length_rod = 10.0
    dist_gap = 0.5
    dist_to_rod = radius_sphere + dist_gap

    k = (4*pi*8.85e-12)**-1

    charge_density_Cpm = -1e-6

    n_field_pts_radial = 8
    n_field_pts_theta = 12
    n_field_pts_phi = 36
    n_source_pts_theta = 32

    radial_max = 0.99 * dist_to_rod
    radial_min = 1.01 * radius_sphere

    radial_field_pts = np.linspace(radial_min, radial_max, n_field_pts_radial)
    theta_field_pts = np.linspace(0.0, 5*pi/16, n_field_pts_theta)
    phi_field_pts = np.linspace(-4*pi/8, 4*pi/8, n_field_pts_phi)


    # potential_array = np.zeros((n_field_pts_radial, n_field_pts_theta, n_field_pts_phi), dtype=float)

    integ_lim_upper = atan(length_rod / dist_to_rod)

    # rodAn in matlab code
    theta_source = np.linspace(0.0, integ_lim_upper, n_source_pts_theta)

    # Do we want matrix or cartesian indexing? Lets go with cartesian...
    mesh_radial_f, mesh_theta_f, mesh_phi_f, mesh_theta_s = np.meshgrid(radial_field_pts, theta_field_pts,
                                                                        phi_field_pts, theta_source)

    def integrand1(rad_f, thet_f, phi_f, thet_s, rad_sph, dist_rod):
        return 1.0 \
            / (np.cos(thet_s) * np.sqrt(rad_f**2 * np.sin(thet_s)**2 * np.cos(phi_f)**2
               + (rad_f * np.sin(thet_f) * np.sin(phi_f) - (rad_sph**2 / dist_rod) * np.cos(thet_s) * np.sin(thet_s))**2
               + (thet_s * np.cos(thet_f) - (rad_sph**2 / dist_rod) * (np.cos(thet_s))**2)**2))

    def integrand2(thet_f, phi_f, thet_s, dist_rod):
        return (1 + np.tan(thet_f)**2) \
            / np.sqrt(thet_s**2 * np.sin(thet_s)**2 * np.cos(thet_s)**2
                      + (thet_s * np.sin(thet_f) * np.sin(phi_f) - dist_rod * np.tan(thet_s))**2
                      + (thet_s * np.cos(thet_f) - dist_rod)**2)


    mesh_potential_V = k * charge_density_Cpm * dist_to_rod/radius_sphere \
        * np.trapz(y=integrand1(rad_f=mesh_radial_f, thet_f=mesh_theta_f, phi_f=mesh_phi_f,
                                thet_s=mesh_theta_s, rad_sph=radius_sphere, dist_rod=dist_to_rod)
                   + integrand2(thet_f=mesh_theta_f, phi_f=mesh_phi_f, thet_s=mesh_theta_s,
                                dist_rod=dist_to_rod),
                   x=theta_source, axis=3)

    potential_V = mesh_potential_V.flatten('F')

    min_max_potential_V = (min(potential_V), max(potential_V))

    mesh_x, mesh_y, mesh_z = spherical_to_cartesian(radial=mesh_radial_f[:, :, :, 0], azimuthal=mesh_theta_f[:, :, :, 0],
                                                    polar=mesh_phi_f[:, :, :, 0])

    x, y, z = [mesh.flatten('F') for mesh in (mesh_x, mesh_y, mesh_z)]

    from matplotlib import pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from mpl_toolkits.mplot3d import Axes3D

    jet = plt.get_cmap('jet')
    c_norm = colors.Normalize(vmin=min_max_potential_V[0], vmax=min_max_potential_V[1])
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)
    color_arr = scalar_map.to_rgba(potential_V)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(xs=x, ys=y, zs=z, c=color_arr)
    plt.show()

    print('whoaaa')

# see ImagSphereEmagVert6.m