from math import pi, atan  # cos, sqrt, sin, tan

import numpy as np
from mayavi import mlab

from utilities.coordinate_utilities import spherical_to_cartesian, cartesian_to_spherical

try:
    from mayavi import engine
except ImportError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()

if __name__ == '__main__':
    radius_sphere = 1.0
    length_rod = 10.0
    dist_gap = 0.5
    dist_to_rod = radius_sphere + dist_gap

    k = (4*pi*8.85e-12)**-1

    charge_density_Cpm = -1e-6

    n_field_pts_radial = 8
    n_field_pts_theta = 36
    n_field_pts_phi = 36
    n_source_pts_theta = 32

    radial_max = 0.9 * dist_to_rod
    radial_min = 1.1 * radius_sphere

    dot_scale = 0.8 * (radial_max - radial_min) / n_field_pts_radial

    radial_field_pts = np.linspace(radial_min, radial_max, n_field_pts_radial)
    theta_field_pts = np.linspace(0.0, 7*pi/16, n_field_pts_theta)
    phi_field_pts = np.linspace(0, 4*pi/8, n_field_pts_phi)


    # potential_array = np.zeros((n_field_pts_radial, n_field_pts_theta, n_field_pts_phi), dtype=float)

    integ_lim_upper = atan(length_rod / dist_to_rod)

    # rodAn in matlab code
    theta_source = np.linspace(0.0, integ_lim_upper, n_source_pts_theta)

    # Do we want matrix or cartesian indexing? Lets go with cartesian...
    mesh_radial_f, mesh_theta_f, mesh_phi_f, mesh_theta_s = np.meshgrid(radial_field_pts, theta_field_pts,
                                                                        phi_field_pts, theta_source, indexing='xy')

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
    # r_len = len(radial_field_pts)
    # t_len = len(theta_field_pts)
    # p_len = len(phi_field_pts)
    # ctr= 0
    # for k in range(p_len):
    #     for i in range(r_len):
    #         for j in range(t_len):
    #
    #             idx = j + i*t_len + k*t_len*r_len
    #
    #             print('flat: {}, mesh: {}'.format(potential_V[idx], mesh_potential_V[j, i, k]))
    #
    #             if potential_V[ctr] != mesh_potential_V[j, i, k]:
    #                 print('nooooooooo')
    #             ctr += 1

    min_max_potential_V = (min(potential_V), max(potential_V))

    mesh_x, mesh_y, mesh_z = spherical_to_cartesian(radial=mesh_radial_f[:, :, :, 0], azimuthal=mesh_theta_f[:, :, :, 0],
                                                    polar=mesh_phi_f[:, :, :, 0])

    x, y, z = [mesh.flatten('F') for mesh in (mesh_x, mesh_y, mesh_z)]

    from matplotlib import pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    jet = plt.get_cmap('jet')
    c_norm = colors.Normalize(vmin=min_max_potential_V[0], vmax=min_max_potential_V[1])
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)
    color_arr = scalar_map.to_rgba(potential_V)
    color_arr.tolist()
    color_arr = [tuple(rgb[:3]) for rgb in color_arr]


    # fig = mlab.figure()


    contour_V = mlab.contour3d(mesh_x, mesh_y, mesh_z, mesh_potential_V)

    # glyphs_V = mlab.points3d(x, y, z, potential_V, scale_mode='none', scale_factor=dot_scale)
    mlab.axes()
    mlab.title('Electric Potential')
    mlab.colorbar()

    mlab.show()

    # x_plane = GridPlane()
    # y_plane = GridPlane()
    # z_plane = GridPlane()
    # x_plane.grid_plane.axis = 'x'
    # y_plane.grid_plane.axis = 'y'
    # z_plane.grid_plane.axis = 'z'
    #
    # if len(engine.scenes) == 0:
    #     engine.new_scene()
    # module_manager = engine.scenes[0].children[0].children[0]
    # module_manager.scalar_lut_manager.scalar_bar.height = 0.8000000000000006
    # module_manager.scalar_lut_manager.scalar_bar.position = array([0.85132568, 0.11616402])
    # module_manager.scalar_lut_manager.scalar_bar.position2 = array([0.11780793, 0.8])
    # module_manager.scalar_lut_manager.scalar_bar.width = 0.11780793319415472
    # module_manager.scalar_lut_manager.scalar_bar.title = 'Volts (V)'
    # module_manager.scalar_lut_manager.scalar_bar.orientation = 'vertical'
    # module_manager.scalar_lut_manager.scalar_bar_representation.position = array([0.85132568, 0.11616402])
    # module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([0.11780793, 0.8])

    # mlab.pipeline.volume(mlab.pipeline.scalar_field(mesh_potential_V))


    # fig.add_(glyphs_V)
    # fig.add_child(x_plane)
    # fig.add_child(y_plane)
    # fig.add_child(z_plane)


# see ImagSphereEmagVert6.m