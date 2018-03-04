from math import pi, atan  # cos, sqrt, sin, tan

import numpy as np

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


    potential_array = np.zeros((n_field_pts_radial, n_field_pts_theta, n_field_pts_phi), dtype=float)

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


    potential_V = k * charge_density_Cpm * dist_to_rod/radius_sphere \
        * np.trapz(y=integrand1(rad_f=radial_field_pts, thet_f=theta_field_pts, phi_f=phi_field_pts,
                                thet_s=theta_source, rad_sph=radius_sphere, dist_rod=dist_to_rod)
                   + integrand2(thet_f=theta_field_pts, phi_f=phi_field_pts, thet_s=theta_source,
                                dist_rod=dist_to_rod),
                   x=theta_source)

# see ImagSphereEmagVert6.m