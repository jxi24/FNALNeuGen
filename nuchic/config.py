""" Class to parse the nuchic input file and to store the settings. """

import shutil

import yaml
from absl import logging

from .utils import make_path
from .histogram import Histogram


class _Settings:
    """ Class to read in the user settings from an input file or
    overwrite from commandline. """

    def __init__(self):
        self.nucleus = None

    def load(self, filename='run.yml'):
        """ Load a settings file. """
        try:
            with open(filename, 'r') as settings_file:
                self.__dict__.update(yaml.safe_load(settings_file))
        except FileNotFoundError:
            shutil.copyfile(make_path('template.yml'), 'run.yml')
            logging.fatal('{} could not be found. '
                          'Creating template at `run.yml` and '
                          'quitting.'.format(filename))

        self.nucleus = None

#    def __getattr__(self, name):
#        return self.__dict__.get(name, False)

    @property
    def settings(self):
        """ Get all settings. """
        return self.__dict__

    def search(self, name):
        """ Search for a given setting. """

    # Run settings
    @property
    def run_settings(self):
        """ Return the dictionary of run settings. """
        return self.__dict__['run']

    @run_settings.setter
    def run_settings(self, name, value):
        """ Set a run setting. """
        self.__dict__['run'][name] = value

    @property
    def nevents(self):
        """ Return the requested number of generated events. """
        return self.run_settings['events']

    @property
    def beam_energy(self):
        """ Return the beam energy. """
        return self.run_settings['beam_energy']

    @property
    def angle(self):
        """ Return the angle of the outgoing electron in degrees. """
        return self.run_settings['angle']

    @property
    def cascade(self):
        """ Return if the cascade should be run. """
        return self.run_settings['cascade']

    @property
    def folding(self):
        """ Return if the folding function should be used. """
        return self.run_settings['folding']

    @property
    def output_format(self):
        """ Get the event output format. """
        return self.run_settings['output']

    # Parameter settings
    @property
    def parameters(self):
        """ Return the dictionary of parameter settings. """
        return self.__dict__['parameters']

    @parameters.setter
    def parameters(self, name, value):
        """ Set a parameter value. """
        self.__dict__['parameters'][name] = value

    @property
    def distance(self):
        """ Maximum propagation distance of particles in cascade. """
        return self.parameters['cascade_distance']

    @property
    def folding_func(self):
        """ Get the user folding function. """
        return self.run_settings['folding_func']

    @property
    def config_type(self):
        """ Return the configuration type.

        Get the configuration type for how to setup the nucleus.
        Current options are either:
            - QMC: Quantum Monte Carlo configuration
            - MF: Mean field configuration
        """
        return self.parameters['config_type']

    # Other settings
    def get_histograms(self):
        """ Build the requested histograms from the yaml file. """
        histograms = {}
        for name, hist in self.__dict__['histograms'].items():
            histograms[name] = Histogram(**hist)

            # TODO: Store information on how to calculate

        return histograms

    # mean free path parameters
    @property
    def mean_free_path(self):
        """ Return the dictionary of mean free path settings. """
        return self.__dict__['mean_free_path']

    @property
    def mfp_xsec(self):
        """ Return the cross-section for the mean free path calculation. """
        return self.mean_free_path['xsec']

    @property
    def mfp_nucleons(self):
        """ Return number of nucleons for the mean free path calculation."""
        return self.mean_free_path['nucleons']

    @property
    def mfp_radius(self):
        """ Return radius of the nucleus for mean free path calculation."""
        return self.mean_free_path['radius']


_SETTINGS = _Settings()


def settings():
    """ Accessor function for the settings. """
    return _SETTINGS
