""" Implement a class for information about the nucleus. """

import re
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from absl import logging
from scipy import interpolate

# from nuChic.particle import Particle
from .constants import MEV, MQE as mN
from .utils import make_path, rand_sphere
from .config import settings

Z_TO_NAME = {
    1: 'H',
    2: 'He',
    3: 'Li',
    6: 'C',
    8: 'O',
    13: 'Al',
    18: 'Ar',
    20: 'Ca',
    26: 'Fe',
}

NAME_TO_Z = {value: key for key, value in Z_TO_NAME.items()}


def main():
    """ Main function for testing speed of the code. """
    import timeit
    print(
        timeit.timeit(
            'n.generate_config()',
            setup="""
                from __main__ import Nucleus;
                from nuChic.Constants import MEV;
                n = Nucleus(6,12,50*MeV, 225*MEV)""",
            number=100) /
        100)


class Nucleus:
    """
    Basic Nucleus class.
    TODO: Flesh out docs
    """
    def __init__(self, Z, A, binding, kf, config_type):
        if Z is None:
            self.protons = settings().mfp_nucleons
            self.neutrons = 0
            self.binding = 0
            self.kf = 0
            self.radius = settings().mfp_radius
            self.potential = 0
            self.density = ((self.protons + self.neutrons)
                            / (4.0/3.0*np.pi*self.radius**3))
            return

        if Z > A:
            raise ValueError('Requires the number of protons be less than '
                             'total number of nucleons. Got {} protons and '
                             '{} nucleons.'.format(Z, A))

        self.protons = Z
        self.nucleons = A
        self.binding = binding
        self.kf = kf
        self.radius = pow(self.nucleons / (4.0 / 3.0 * np.pi *
                                           0.16), 1.0 / 3.0)
        self.potential = (np.sqrt(mN**2 + self.kf**2) - mN + 8*MEV)

        # If it is Carbon-12, let's use Noemi's configurations.
        # We read it when defining the nucleus and then we only need to pick a
        # configuration when running the cascade
        # 1 million configurations, no header
        # index   pid    x    y    z
        if Z == 6 and A == 12:
            self.density = pd.read_csv(
                make_path('{}_configs.out.gz'.format(config_type.upper()),
                          'configurations'),
                sep=r'\s+',
                names=['pid', 'x', 'y', 'z'],
                compression='gzip'
            )
        else:
            raise NotImplementedError('The nucleus with {} protons and {} '
                                      'neutrons is currently not '
                                      'implemented.'.format(Z, Z-A))

        test_density = pd.read_csv('~/Documents/Projects/NeutrinoGenerator/FNALNeuGen/densities/c12.density', header=10,
                sep='\s+', names=['r', 'rho', 'error'])
        self.test_density = interpolate.InterpolatedUnivariateSpline(test_density['r'].values,
                test_density['rho'], k=1)

    @staticmethod
    def make_nucleus(name, binding, kf, config_type):
        """ Generate a nucleus from a string. """
        logging.info('Initializing Nucleus: Found nucleus {}'.format(name))
        match = re.match(r'([0-9]+)([a-zA-Z]+)', name, re.I)
        if match:
            nucleons, protons = match.groups()
            nucleons = int(nucleons)
            protons = NAME_TO_Z[protons.upper()]

            return Nucleus(protons, nucleons, binding, kf, config_type)

        # Testing of mean free path for cascade
        if name == 'mfp_calc':
            return Nucleus(None, None, None, 0, None)

        raise ValueError('Invalid nucleus {}.'.format(name))

    def __str__(self):
        """ Convert nucleus into a string, written in nuclear notation. """
        return str(self.nucleons) + Z_TO_NAME[self.protons]

    def escape(self, particle):
        """Check whether or not the particle escaped.
        If the particle escapes, the spatial components of the momentum are
        updated to account for the influence of the potential.
        """
        if particle.status == -2:
            return True
        # # Is the position still within the nucleus?
        # if particle.pos.mag < self.radius:
        #     return False
        # Is the particle moving inwards or outwards?
        # if np.any(np.abs(np.sign(particle.pos.array) - np.sign(particle.mom.array[1:])) > 1):
        #     return False
        # Is the "kinetic energy" less than the binding potential?
        energy_total = np.sqrt(particle.mom.mom2 + particle.mom.mass2)
        energy_kinetic = energy_total - particle.mom.mass
        if energy_kinetic < self.potential:
            return False
        # Then the particle has escaped. Update the momentum to account for
        # the effect of the potential
        theta = particle.mom.theta
        phi = particle.mom.phi
        particle.mom.p_x -= self.potential * np.sin(theta) * np.cos(phi)
        particle.mom.p_y -= self.potential * np.sin(theta) * np.sin(phi)
        particle.mom.p_z -= self.potential * np.cos(theta)
        return True

    def absorb(self, particle):
        """TODO Implement absorb, add doc"""


#    def generate_config(self):
#        def to_cartesian(coords):
#            #r, theta, phi = coords
#            r = coords[:,0]
#            theta = coords[:,1]
#            phi= coords[:,2]
#            x = r*np.sin(theta)*np.sin(phi)
#            y = r*np.sin(theta)*np.cos(phi)
#            z = r*np.cos(theta)
#            return np.transpose(np.array([x, y, z]))
#
#        protons = np.random.random(self.Z*3)
#        protons = protons.reshape(self.Z,3)
#        protons[:,0] = protons[:,0]*self.radius
#        protons[:,1] = np.arccos(2*protons[:,1] - 1)
#        protons[:,2] = protons[:,2]*2*np.pi
#
#        neutrons = np.random.random((self.A-self.Z)*3)
#        neutrons = neutrons.reshape((self.A-self.Z),3)
#        neutrons[:,0] = neutrons[:,0]*self.radius
#        neutrons[:,1] = np.arccos(2*neutrons[:,1] - 1)
#        neutrons[:,2] = neutrons[:,2]*2*np.pi
#
#        protons = to_cartesian(protons)
#        neutrons = to_cartesian(neutrons)
#
#        return protons, neutrons

    def generate_config(self):
        """ Generate configurations based off the density profile. """

        # Given a set of configurations to choose from
        if isinstance(self.density, pd.DataFrame):
            config_index = np.random.randint(
                0, high=len(self.density.index) / 12)
            idx0 = config_index * 12

            db_tmp = self.density.iloc[idx0:idx0 + 12]
            proton_mask = db_tmp['pid'] == 1
            neutron_mask = db_tmp['pid'] == -1
            protons = np.asarray(db_tmp[proton_mask][['x', 'y', 'z']])
            neutrons = np.asarray(db_tmp[neutron_mask][['x', 'y', 'z']])

            # Rotations using Euler angles in the "x-convention"
            angles = np.random.random(3) * 2 * np.pi
            angles[1] /= 2.
            rotation = Rotation.from_euler('zxz', angles)

            protons = rotation.apply(protons)
            neutrons = rotation.apply(neutrons)

            return protons, neutrons

        # Given function to generate configurations from
        if callable(self.density):
            raise NotImplementedError(
                'Function for density currently not implemented.')

        # Given a constant density to generate from
        if isinstance(self.density, float):
            num_pts = self.protons + self.neutrons
            nucleons = rand_sphere(self.radius, num_pts)
            return nucleons[:self.protons], nucleons[self.protons:]

    def generate_momentum(self):
        """Generate a random momentum 3-vector in cartesian coordinates."""
        def to_cartesian(coords):
            """Convert spherical coordinates to cartesian coordinates."""
            # r, theta, phi = coords
            r = coords[0]
            theta = coords[1]
            phi = coords[2]
            x = r * np.sin(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.cos(phi)
            z = r * np.cos(theta)
            return np.transpose(np.array([x, y, z]))

        momentum = np.random.random(3)
        # Random numbers fist to spherical coords
        momentum[0] = momentum[0] * self.kf
        momentum[1] = np.arccos(2 * momentum[1] - 1)
        momentum[2] = momentum[2] * 2 * np.pi
        # and then to cartesian
        momentum = to_cartesian(momentum)
        return momentum


if __name__ == '__main__':
    main()
