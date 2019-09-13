#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Implements the Cascade calculation """

import numpy as np

from nuChic.four_vector import Vec4
from nuChic.ThreeVector import Vec3
from nuChic.Particle import Particle
from nuChic.Nucleus import Nucleus
from nuChic.constants import FM as fm, MQE as mN
from nuChic.Interaction import sigma_pp, sigma_np

from absl import logging


class FSI:
    """ Notes to self:
    * Initialize nucleus
      - Config of p,n
      - Has as input the “kick” (which particle, how much)
    * “Main cascade” functionality
        ** Reabsorption
        ** Exiting
        Notes
        -----
        TODO: Implement binding energy / nuclear potential
        TODO: Implement realistic:
              - initial state configurations (position and momentum)
              - cross sections (distinguishing n from p)
              - reabsorption routine
    """

    def __init__(self, nucleus, dt):
        """
        Generates nucleus configuration and kicked nucleon.

        Args:
            nucleus: nuChic.Nucleus
            energy_transfer: float, energy transfered to the nucleus.
            dt: float, time step
        """
        # if not isinstance(nucleus, Nucleus):
        #    raise TypeError('Expected a Nucleus as input, got {}'
        #                    % type(nucleus))
        self.nucleus = nucleus
        self.time_step = dt

        # Generate p,n position distribution
        protons, neutrons = self.nucleus.generate_config()

        # Define particles from configuration
        # NOTE: their 4-momentum is not physical right now
        dummy_mom = Vec4(mN, np.nan, np.nan, np.nan)
        self.nucleons = [
            Particle(
                pid=2212, mom=dummy_mom, pos=Vec3(*x_j)
            ) for x_j in protons
        ]
        self.nucleons += [Particle(pid=2112, mom=dummy_mom,
                                   pos=Vec3(*x_j)) for x_j in neutrons]

        # Keep outgoing particles after cascade
        # self.outgoing_particles = []

        # Cylinder parameters
        self.cylinder_pt1 = 0
        self.cylinder_pt2 = 0

        self.kicked_idxs = []

    @property
    def number_nucleons(self):
        """ Returns the number of nucleons in the nucleus """
        return len(self.nucleons)

    @property
    def number_protons(self):
        """ Returns the number of protons in the nucleus """
        count = 0
        for particle in self.nucleons:
            if particle.pid == 2212:
                count += 1
        return count

    @property
    def number_neutrons(self):
        """ Returns the number of neutrons in the nucleus """
        count = 0
        for particle in self.nucleons:
            if particle.pid == 2112:
                count += 1
        return count

    def kick(self, energy_transfer):
        '''
        Randomize kicked particle

        Args:
            energy_transfer: float, energy transfered to the nucleus
        '''
        self.kicked_idxs = []
        self.kicked_idxs.append(np.random.randint(
            low=0, high=len(self.nucleons)))
        self.nucleons[self.kicked_idxs[0]].status = -1  # propagating nucleon
        # Ep = mN+energy_transfer
        # pp = np.sqrt(Ep**2-mN**2)
        # self.nucleons[self.kicked_idxs[0]].mom = Vec4(Ep, 0, 0, pp)
        self.nucleons[self.kicked_idxs[0]].mom = energy_transfer

    def reset(self):
        '''
        Reset the FSI parameters to begin the next calculation
        '''
        # Generate p,n position distribution
        protons, neutrons = self.nucleus.generate_config()

        # Define particles from configuration
        # NOTE: their 4-momentum is not physical right now
        dummy_mom = Vec4(mN, np.nan, np.nan, np.nan)
        self.nucleons = [
            Particle(
                pid=2212, mom=dummy_mom, pos=Vec3(*x_j)
            ) for x_j in protons
        ]
        self.nucleons += [Particle(pid=2112, mom=dummy_mom,
                                   pos=Vec3(*x_j)) for x_j in neutrons]

        # Keep outgoing particles after cascade
        # self.outgoing_particles = []

        # Cylinder parameters
        self.cylinder_pt1 = 0
        self.cylinder_pt2 = 0

    def __call__(self):
        ''' Performs the full propagation of the kicked nucleons inside
        the nucleus. Updates the list of outgoing_particles with
        all status=+1 particles
        '''
        # Overestimate cross section
        sigma = 100*fm**2  # 0.1 barn xsec = 10 fm^2
        # positions = []
        # positions_temp = []
        for step in range(10000):
            logging.debug('*******  STEP ', step, ' *******')
            if self.kicked_idxs == []:
                logging.debug('No more particles propagating - DONE!')
                break
            # Update formation zones
            for i in range(len(self.nucleons)):
                if self.nucleons[i].is_in_formation_zone():
                    self.nucleons[i].formation_zone -= self.time_step
            # copy to avoid changing during iteration
            new_kicked_idxs = list(self.kicked_idxs)
            for kick_idx in self.kicked_idxs:
                logging.debug('kick_idx = ', kick_idx)
                did_hit, new_kick_idx = self.interacted(kick_idx, sigma)
                if did_hit:
                    logging.debug('Hit?')
                    (really_did_hit, self.nucleons[kick_idx],
                     self.nucleons[new_kick_idx]) = \
                        self.generate_final_phase_space(
                            self.nucleons[kick_idx],
                            self.nucleons[new_kick_idx]
                        )
                    # if it really hit, add index to new kicked index
                    # list and delete duplicates
                    if really_did_hit:
                        logging.debug('Hit!!!!')
                        new_kicked_idxs.append(new_kick_idx)
                        new_kicked_idxs = list(
                            set(new_kicked_idxs))  # Remove duplicates

#                    # Is it a background particle?
#                    # If so, we need to generate it's momentum
#                    if not self.nucleons[new_kick_idx].is_propagating() :
#                        # Sort background particle 4-momentum
#                        p_i = Vec3(*self.nucleus.generate_momentum())
#                        energy = np.sqrt(mN**2+p_i.P2())
#                        p_mu = Vec4(energy, *p_i.Vec())
#                        self.nucleons[new_kick_idx].mom=p_mu
#                        # Hit background nucleon becomes propagating nucleon
#                        self.nucleons[new_kick_idx].status=-1
#                        # Add it to list of kicked particles
#                        new_kicked_idxs.append(new_kick_idx)
#                    # Generating outgoing phase space
#                    p1_out, p2_out = self.generate_final_phase_space(
#                                       self.nucleons[kick_idx],
#                                       self.nucleons[new_kick_idx])
                    # Check for Pauli blocking
#                    if self.pauli_blocking(p1_out) or
#                       self.pauli_blocking(p2_out):
                        # Pauli blocking occurred, revert to old configuration

#                else:
#                    logging.debug('No hit')

            self.kicked_idxs = new_kicked_idxs
            logging.debug('kicked_idxs = ', self.kicked_idxs)

            # After-hit checks
            not_propagating = []
        #    for i in range(len(kicked_idxs)):
            for i, kick_idx in enumerate(self.kicked_idxs):
                #        kick_idx = kicked_idxs[i]
                # Nucleon becomes final particle if
                # (1) is outside nucleus or
                if self.nucleons[kick_idx].pos.P() > self.nucleus.radius:
                    not_propagating.append(i)
                    if self.nucleus.escape(self.nucleons[kick_idx]):
                        self.nucleons[kick_idx].status = 1
                        logging.debug('nucleon {} is OOOOOOUT! \
                                       status: {}'.format(
                                           kick_idx,
                                           self.nucleons[kick_idx].status)
                                      )
                    else:
                        self.nucleons[kick_idx].status = 2
                        logging.debug('nucleon {} is captured! \
                                       status: {}'.format(
                                           kick_idx,
                                           self.nucleons[kick_idx].status)
                                      )
                # (2) has kinetic energy below some barrier energy
#               elif (self.nucleons[kick_idx].E()-mN < 30*MeV):
#                   not_propagating.append(i)
#                   self.nucleons[kick_idx].status=0
#                   if VERBOSE:
#                       logging.debug('nucleon ', kick_idx,'
#                                     'is reabsorbed! status: ',
#                             self.nucleons[kick_idx].status)
            # Delete indices of non-propagating particles.
            # Delete in reverse order to avoid shifting elements.
            for i in sorted(not_propagating, reverse=True):
                del self.kicked_idxs[i]

            # Save positions for 3D plot
#            for jj in range(len(self.nucleons)) :
#                positions_temp.append(self.nucleons[jj].pos.Vec())
#            positions.append(positions_temp)
#            positions_temp=[]

#            stat_list = [n.status for n in self.nucleons]
#            logging.debug('All status: ', stat_list)

        #     if did_hit:
        #         logging.debug('idxs:  ', kicked_idxs, new_kick_idx )
        #         logging.debug('out momenta: ', mom1, mom2)
        #         logging.debug('sanity check',
        #                       mom1-nucleons[kick_idx].mom,
        #                       mom2-nucleons[new_kick_idx].mom)
#        logging.debug('Number of steps: ',step)
        stat_list = [n.status for n in self.nucleons]
#        logging.debug('All status: ', stat_list)
#        logging.debug('Number of final state nucleons: ',sum(stat_list))
        if -1 in stat_list:
            logging.fatal(
                "Cascade Failed at step: {}, ",
                "has at least one propagating nucleon still" % step)

        # Record outgoing particles
        # self.outgoing_particles = [n for n in self.nucleons if n.is_final()]

        return [n for n in self.nucleons if n.is_final()]

    @staticmethod
    def points_in_cylinder(pt1, pt2, radius, position):
        ''' Check if a point is within a cylinder
        Args:
            pt1: initial position vector
            pt2: final position vector
            radius: radius of cylinder
            position: position vector of particle in question
        '''
        pt1 = np.asarray(pt1)
        pt2 = np.asarray(pt2)
        position = np.asarray(position)
        vec = pt2 - pt1
        const = radius * np.linalg.norm(vec)
        return (np.dot(position - pt1, vec) >= 0
                and np.dot(position - pt2, vec) <= 0
                and np.linalg.norm(np.cross(position - pt1, vec)) <= const)

    @staticmethod
    def to_cartesian(coords):
        ''' Takes spherical coordinates [r, theta, phi] and transform to
        cartesian coordinates [x,y,z]
        '''
        # radius, theta, phi = coords
        radius = coords[0]
        theta = coords[1]
        phi = coords[2]
        pos_x = radius*np.sin(theta)*np.sin(phi)
        pos_y = radius*np.sin(theta)*np.cos(phi)
        pos_z = radius*np.cos(theta)
        return np.transpose(np.array([pos_x, pos_y, pos_z]))

    def interacted(self, idx, sigma):
        ''' Decides if an interaction occurred for a propagating particle
        within given time step

            Parameters
            ----------
                idx : int
                    Index of propagating particle.
                sigma : float
                    Nucleon-nucleon scattering cross section in fm^2

            Returns
            -------
            (key , i), with
            key : bool
                Whether the interaction occured
            i : int
                Particle label

            Notes
            -----
            TODO: This needs to be improved to take into account different
                  cross sections for different nucleai). Maybe this can be
                  an overestimate and the real cross section will be checked in
                  generate_final_phase_space?
        '''

        # Builds up cylinder
        self.cylinder_pt1 = self.nucleons[idx].pos
        self.nucleons[idx].propagate(self.time_step)
        self.cylinder_pt2 = self.nucleons[idx].pos
        cylinder_r = np.sqrt(sigma/np.pi)
#        key = False
        # Check if any particle (except propagating one) is within cylinder
        # Stops when first is found (not closest one)
        # TODO: optimize with self.n_particles?
        idxs = np.arange(len(self.nucleons))
        in_cylinder = False
        if self.nucleons[idx].is_in_formation_zone():
            return False, np.nan
        for i in idxs[np.where(idxs != idx)]:
            if self.nucleons[i].is_final() or \
                    self.nucleons[i].is_in_formation_zone():
                continue
            position = self.nucleons[i].pos.Vec()
            in_cylinder = self.points_in_cylinder(
                self.cylinder_pt1.array(),
                self.cylinder_pt2.array(),
                cylinder_r,
                position
            )
            if in_cylinder:
                # Found particle in cylinder
                return in_cylinder, i
        return False, None

    def generate_final_phase_space(self, particle1, particle2):
        ''' Generates phase space (isotropic), checks if particle is inside
        cylinder for proper cross section, checks for Pauli blocking
        (if yes, revert to inital state), set formation zones for
        scattered particles

            Parameters
            ----------
                particle1in : Particle
                    Interating particle 1
                particle2in : Particle
                    Interating particle 2

            Returns
            -------
            (really_did_hit, particle1, particle2), with
            really_did_hit : bool
                False if Pauli blocking occurs (interation did not happen)
            particle1 : Particle
                Outgoing particle 1
            particle2 : Particle
                Outgoing particle 2

            Notes
            -----
            Calls initial phase space generation, gives back fully update
            particles with their corresponding status and formation zones
            (or input particles in case of Pauli blocking)
            TODO: Implement inelastic scattering
            TODO: Discriminate pp, np, nn scatterings (phase space)
            TODO: Implement realistic phase space
        '''

        # We do not want to change the input particles in case
        # Pauli blocking occurred
#        particle1 = deepcopy(particle1in)
#        particle2 = deepcopy(particle2in)

        # Is particle 2 a background particle? If so, we need to
        # generate it's momentum
        if particle2.is_background():
            # Sort background particle 4-momentum
            p_i = Vec3(*self.nucleus.generate_momentum())
            energy = np.sqrt(mN**2+p_i.P2())
            p_mu = Vec4(energy, *p_i.Vec())
            particle2.mom = p_mu

        # Start generation of final state phase space
        # Boost back to CoM frame
        total_momentum = particle1.mom+particle2.mom
        boost_vec = total_momentum.BoostVector()
#        cm_momentum = total_momentum.BoostBack(boost_vec)
        # Fully elastic scattering, protons and neutrons
        # are being treated equally
        # TODO: Inelastic scattering

        # Calculate if particles are inside cylinder with proper momentum
        # dependent cros section.
        # First, we need the momentum of particle 1 in the lab frame.
        boost_vec_lab = particle2.mom.BoostVector()
        lab_particle1_momentum = particle1.mom.BoostBack(boost_vec_lab).P()
#        lab_particle1_momentum = lab_particle1_momentum.P()
        # Calculate cross section (flavor dependent)
        if particle1.pid == particle2.pid:
            sigma_p_dependent = sigma_pp(lab_particle1_momentum)
        else:
            sigma_p_dependent = sigma_np(lab_particle1_momentum)
        # Check cylinder:
        # (using global cylinder variables defined in interacted)
        cylinder_r = np.sqrt(sigma_p_dependent/np.pi)
        position = particle2.pos.Vec()
        really_did_hit = self.points_in_cylinder(
            self.cylinder_pt1.array(),
            self.cylinder_pt2.array(),
            cylinder_r,
            position
        )
        if not really_did_hit:
            return really_did_hit, particle1, particle2

        # The original frame is not the lab frame, since both particles have,
        # in general, non-zero momenta
        # Particle 4-momentum in CoM frame
        p1_cm = particle1.mom.BoostBack(boost_vec)
        p2_cm = particle2.mom.BoostBack(boost_vec)

        # Fix magnitude, (theta, phi) generated isotropically
        momentum = np.random.random(3)
        momentum[0] = p1_cm.P()
        momentum[1] = np.arccos(2*momentum[1] - 1)
        momentum[2] = momentum[2]*2*np.pi

        # Three momentum in cartesian coordinates:
        momentum = self.to_cartesian(momentum)

        # Outgoing 4-momenta
        p1_out = Vec4(*[p1_cm.E, *momentum])
        p2_out = Vec4(*[p1_cm.E, *(-momentum)])

        # Get q0 in lab frame for formation zone
        # TODO: what happens in inelastic scatterings?
        q_lab = p2_out.BoostBack(p2_cm.BoostVector()) - \
            p2_cm.BoostBack(p2_cm.BoostVector())

        # Boost to original frame
        p1_out = p1_out.Boost(boost_vec)
        p2_out = p2_out.Boost(boost_vec)

        # Check for Pauli blocking and return initial particles if it occurred
        really_did_hit = not(self.pauli_blocking(
            p1_out) or self.pauli_blocking(p2_out))
        if really_did_hit:
            # Assign momenta to particles
            particle1.mom = p1_out
            particle2.mom = p2_out

            # Assign formation zone
            t = q_lab.M2()
            particle1.set_formation_zone(q_lab.E, t, 0.139)
            particle2.set_formation_zone(q_lab.E, t, 0.139)
            # logging.debug("form zone = ",foo, q_lab.E, t)

            # Hit background nucleon becomes propagating nucleon
            particle2.status = -1

        return really_did_hit, particle1, particle2
#        else :
#            return really_did_hit, particle1, particle2in
#            particle1 = deepcopy(particle1in)
#            particle2 = deepcopy(particle2in)

    def pauli_blocking(self, four_momentum):
        ''' Checks if Pauli blocking occurred

            Parameters
            ----------
                four_momentum: Fourvector
                    Fourvector of particle in question

            Returns
            -------
            True if Pauli blocking occurred

            Notes
            -----
            Right now, this only checks if magnitude of 3-momentum is
            below Fermi motion
        '''
        # See if Pauli blocking occurs for the proposed interaction
        if four_momentum.P() < self.nucleus.kf:
            return True
        return False