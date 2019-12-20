from ast import literal_eval
import os

from absl import flags, app, logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit

from .vegas import Integrator
from .inclusive import Quasielastic
from .cascade import FSI

from .constants import FM, GEV, MQE
from .utils import momentum_sort
from .histogram import Histogram
from .nucleus import Nucleus
from .config import settings
from .qe_data import QuasielasticData
from .particle import Particle
from .four_vector import Vec4
from .three_vector import Vec3

FLAGS = flags.FLAGS
flags.DEFINE_bool(
    'cascade', None, 'Flag to turn on/off the cascade', short_name='c')
flags.DEFINE_bool('folding', None, 'Flag to turn on/off the folding function')
flags.DEFINE_bool('cascade_test', None, 'Flag to test the cascade code')
flags.DEFINE_bool('mfp_test', None, 
                  'Flag to test mean free path of cascade algorithm')
flags.DEFINE_bool('testing', None, 'Flag to test the basics of the program')


# TODO: Figure out how to implement this correctly
def init_histograms(filename='hists.txt'):
    """ Load the desired histograms to be plotted from input file. """
    histograms = {}
    with open(filename) as hist_file:
        for line in hist_file:
            print(line)
            tmp = literal_eval(line)
            histograms[tmp.name] = tmp


class NuChic:
    """ Main driver class for the nuChic code. """
    def __init__(self, run_card):
        settings().load(run_card)

        nucleus = Nucleus.make_nucleus(
            settings().__dict__['run']['nucleus'],
            settings().__dict__['parameters']['binding_energy'],
            settings().__dict__['parameters']['fermi_momentum'],
            settings().__dict__['parameters']['config_type'])
        settings().nucleus = nucleus

        self.histograms = {'omega': Histogram([0, 0.6], 200),
                           'omega2': Histogram([0, 0.6], 200),
                           'px_pre': Histogram([-500, 500], 100),
                           'py_pre': Histogram([-500, 500], 100),
                           'pz_pre': Histogram([-500, 500], 100),
                           'e_pre': Histogram([500, 1500], 100),
                           'px_post': Histogram([-500, 500], 100),
                           'py_post': Histogram([-500, 500], 100),
                           'pz_post': Histogram([-500, 500], 100),
                           'e_post': Histogram([500, 1500], 100),
                           'px_diff': Histogram([-500, 500], 100),
                           'py_diff': Histogram([-500, 500], 100),
                           'pz_diff': Histogram([-500, 500], 100),
                           'e_diff': Histogram([-500, 500], 100),
                           'escape': Histogram([-0.5, 12.5], 13),
                           'escape_proton': Histogram([-0.5, 12.5], 13),
                           'pt_miss': Histogram([-500, 500], 100),
                           'wgts': Histogram(bins=np.logspace(-4, 4, 400)),
                           }

        self.inclusive = Quasielastic(0,
                                      settings().beam_energy,
                                      settings().angle)
        if FLAGS.cascade:
            self.fsi = FSI(settings().distance*FM)

        self.nevents = int(settings().nevents)

    def run(self):
        """ Runs the nuChic code with given settings."""

        if FLAGS.cascade_test:
            self.calc_cross_section()
            return

        if FLAGS.mfp_test:
            self.calc_mfp()
            return

        ndims = 3
        if FLAGS.folding:
            ndims += 1

        integ = Integrator([[0, 1]]*ndims)
        result = integ(self.generate_one_event, nitn=10, neval=1e4)
        print(result.summary())

        zeros = 0

        points, weights, _ = integ.generate(self.nevents)
        for i in tqdm(range(self.nevents), ncols=80):
            if self.generate_one_event(points[i], weights[i]) == 0:
                zeros += 1

        print(zeros)
        self.plot_results()

    def generate_one_event(self, point, weight=None):
        """ Generate the output for a single event. """

        wgt, variables = self.inclusive.generate_weight(point)
        if weight is None:
            return np.sum(wgt)

        if wgt.all() == 0:
            return 0

        momentum = self.inclusive.generate_momentum(variables)

        self.fill_hists(momentum, variables, wgt*weight)

        return np.sum(wgt)*weight

    def fill_hists(self, momentum, variables, lwgt):
        """ Fill the histograms. """

        wgt = np.sum(lwgt)
        if momentum is not None:
            self.histograms['omega'].fill(variables.omega / GEV, wgt)
            self.histograms['e_pre'].fill(momentum.energy, wgt)
            self.histograms['px_pre'].fill(momentum.p_x, wgt)
            self.histograms['py_pre'].fill(momentum.p_y, wgt)
            self.histograms['pz_pre'].fill(momentum.p_z, wgt)
            self.histograms['wgts'].fill(wgt)

            if FLAGS.cascade:
                self.fsi.kick(momentum, lwgt)
                escaped = self.fsi()
                escaped_protons = [particle for particle in escaped
                                   if particle.pid == 2212]
                self.histograms['escape'].fill(len(escaped), wgt)
                self.histograms['escape_proton'].fill(len(escaped_protons),
                                                      wgt)
                self.fsi.reset()

                if len(escaped_protons) == 1:
                    e_beam = settings().beam_energy
                    e_prime = e_beam - variables.omega
                    angle = settings().angle
                    pt_electron = np.abs(e_prime*np.sin(angle))
                    norm = variables.qval
                    rot_mat = np.array([[e_beam - e_prime*np.cos(angle),
                                         0,
                                         -e_prime*np.sin(angle)],
                                        [0, norm, 0],
                                        [e_prime*np.sin(angle),
                                         0,
                                         e_beam - e_prime*np.cos(angle)]])
                    rot_mat /= norm
                    p_proton = escaped_protons[0].mom
                    p_proton.set_vec3(p_proton.vec3.rotate(rot_mat))
                    pt_proton = p_proton.p_t
                    self.histograms['pt_miss'].fill(pt_proton - pt_electron,
                                                    wgt)

                if escaped:
                    self.histograms['omega2'].fill(variables.omega / GEV, wgt)
                    escaped.sort(reverse=True, key=momentum_sort)
                    momentum_post = escaped[0].mom

                    self.histograms['e_post'].fill(momentum_post.energy, wgt)
                    self.histograms['px_post'].fill(momentum_post.p_x, wgt)
                    self.histograms['py_post'].fill(momentum_post.p_y, wgt)
                    self.histograms['pz_post'].fill(momentum_post.p_z, wgt)

                    self.histograms['e_diff'].fill(
                        momentum_post.energy - momentum.energy, wgt)
                    self.histograms['px_diff'].fill(
                        momentum_post.p_x - momentum.p_x, wgt)
                    self.histograms['py_diff'].fill(
                        momentum_post.p_y - momentum.p_y, wgt)
                    self.histograms['pz_diff'].fill(
                        momentum_post.p_z - momentum.p_z, wgt)

    def plot_results(self):
        """ Plot results of the run. """
        qe_data = QuasielasticData(str(settings().nucleus))
        energy = settings().beam_energy / GEV
        angle = settings().angle

        data = qe_data.get_data(energy, angle)

        _, ax1 = plt.subplots(nrows=1, ncols=1)
        self.histograms['omega'].plot(ax1)
        self.histograms['omega2'].plot(ax1)
        ax1.errorbar(data[0], data[1],
                     yerr=data[2], fmt=',', color='red',
                     capsize=2.5)

        plt.savefig('omega.png', bbox_inches='tight')

        fig3, axs2 = plt.subplots(nrows=2, ncols=2)
        self.histograms['e_pre'].plot(axs2[0][0])
        self.histograms['px_pre'].plot(axs2[0][1])
        self.histograms['py_pre'].plot(axs2[1][0])
        self.histograms['pz_pre'].plot(axs2[1][1])
        fig3.suptitle('Before Cascade')

        plt.savefig('before_cascade.png', bbox_inches='tight')

        if FLAGS.cascade:
            fig2, axs = plt.subplots(nrows=2, ncols=2)
            self.histograms['e_post'].plot(axs[0][0])
            self.histograms['px_post'].plot(axs[0][1])
            self.histograms['py_post'].plot(axs[1][0])
            self.histograms['pz_post'].plot(axs[1][1])
            fig2.suptitle('After Cascade')

            plt.savefig('after_cascade.png', bbox_inches='tight')

            fig4, axs3 = plt.subplots(nrows=2, ncols=2)
            self.histograms['e_diff'].plot(axs3[0][0])
            self.histograms['px_diff'].plot(axs3[0][1])
            self.histograms['py_diff'].plot(axs3[1][0])
            self.histograms['pz_diff'].plot(axs3[1][1])
            fig4.suptitle('Difference in momentum')

            plt.savefig('difference_cascade.png', bbox_inches='tight')

            _, ax8 = plt.subplots(nrows=1, ncols=1)
            self.histograms['escape'].plot(ax8)
            ax8.set_yscale('log')
            plt.savefig('escape.png', bbox_inches='tight')

            _, ax9 = plt.subplots(nrows=1, ncols=1)
            self.histograms['escape_proton'].plot(ax9)
            ax9.set_yscale('log')
            plt.savefig('escape_proton.png', bbox_inches='tight')

            _, ax10 = plt.subplots(nrows=1, ncols=1)
            self.histograms['pt_miss'].plot(ax10)
            plt.savefig('pt_miss.png', bbox_inches='tight')

    def calc_cross_section(self):
        """ Calculate the pC and nC cross-sections

        Returns: pC and nC cross-sections

        """
        # Initialize FSI
        self.fsi = FSI(settings().distance*FM, test_cascade=True)

        # Initialize variables
        radius = 20*FM
        nscatter_p = 0
        nscatter_n = 0
        steps = 8000

        # Loop over events
        for _ in tqdm(range(self.nevents), ncols=80):
            while True:
                position = radius*(2*np.random.rand(2)-1)
                if np.sum(position**2) < radius**2:
                    break

            position = Vec3(position[0],
                            position[1],
                            -2.7)
            momentum = Vec4(np.sqrt(MQE**2 + settings().beam_energy**2),
                            0, 0,
                            settings().beam_energy)

            # Perform proton calculation
            proton = Particle(2212, momentum, position)
            self.fsi.kicked_idxs = [len(self.fsi.nucleons)]
            self.fsi.nucleons = np.append(self.fsi.nucleons, proton)
            self.fsi.nucleons[-1].status = -2
            status = self.fsi(steps)
            if status == 1:
                nscatter_p += 1.0
            self.fsi.reset()

            # Perform neutron calculation
            # neutron = Particle(2112, momentum, position)
            # self.fsi.kicked_idxs = [len(self.fsi.nucleons)]
            # self.fsi.nucleons.append(neutron)
            # self.fsi.nucleons[-1].status = -1
            # self.fsi(steps)
            # if self.fsi.scatter:
            #     nscatter_n += 1
            # self.fsi.reset()

        print(nscatter_p)
        xsec = np.pi*radius**2/float(self.nevents)*np.array(nscatter_p)
        print(np.pi*radius**2)
        print(xsec.tolist())
        return xsec.tolist()

    def calc_mfp(self):
        """ Calculate the mean free path of the cascade algorithm. """

        # Initialize FSI
        self.fsi = FSI(settings().distance*FM, mfp=True)

        # Initialize variables
        distances = []
        steps = int(settings().nucleus.radius/settings().distance)

        # Loop over events
        for _ in tqdm(range(self.nevents), ncols=80):
            position = Vec3(0, 0, 0)
            momentum = Vec4(np.sqrt(MQE**2 + settings().beam_energy**2),
                            0, 0,
                            settings().beam_energy)

            proton = Particle(2212, momentum, position)
            self.fsi.kicked_idxs = [len(self.fsi.nucleons)]
            self.fsi.nucleons = np.append(self.fsi.nucleons, proton)
            output = self.fsi(steps)
            if not isinstance(output, list):
                distances.append(output)
            self.fsi.reset()

        mean_free_path = (settings().mfp_xsec*settings().nucleus.density)**-1
        lam = np.linspace(0, settings().nucleus.radius, steps)
        theory = 1.0/mean_free_path*np.exp(-lam/mean_free_path)

        counts, bins = np.histogram(distances, bins=lam)

        error = np.sqrt(counts)/np.sum(counts*np.diff(bins))

        _, axis = plt.subplots()

        counts, bins, _ = axis.hist(distances, bins=lam,
                                    density=True, label='data',
                                    histtype='step',
                                    align='mid')

        error_up = np.insert(counts + error, 0, counts[0]+error[0])
        error_dw = np.insert(counts - error, 0, counts[0]-error[0])
        plt.fill_between(bins, error_up, error_dw,
                         alpha=0.3, color=u'#1f77b4',
                         step='pre')

        nonzero = np.nonzero(counts)

        xdata = (bins[1:] + bins[:-1])/2.0
        counts = counts[nonzero]
        xdata = xdata[nonzero]
        error = error[nonzero]

        init = [mean_free_path]

        def fitfunc(point, param):
            return 1.0/param*np.exp(-point/param)

        popt, pcov = curve_fit(fitfunc, xdata, counts,
                               p0=init,
                               sigma=error)
        perr = np.sqrt(pcov[0][0])
        print('Number of multiple nucelons in cylinder = {}'.format(
            self.fsi.multiple_nucleons))
        print('Theory mfp: {:4f}'.format(mean_free_path))
        print('Fit result: {:4f} +/- {:4f}'.format(popt[0],
                                                   perr))

        axis.text(0.83, 0.87, 'Theory mfp: {:4f}'.format(mean_free_path),
                  horizontalalignment='left',
                  verticalalignment='center', transform=axis.transAxes)
        axis.text(0.83, 0.84, 'Fit result: {:4f} +/- {:4f}'.format(popt[0],
                                                                   perr),
                  horizontalalignment='left',
                  verticalalignment='center', transform=axis.transAxes)

        plt.plot(lam, fitfunc(lam, popt), label='fit', color='orange')
        plt.fill_between(lam, fitfunc(lam, popt+np.sqrt(pcov[0][0])),
                         fitfunc(lam, popt-np.sqrt(pcov[0][0])),
                         color='orange',
                         alpha=0.5)
        plt.plot(lam, theory, label='theory', color='green')
        plt.legend()
        plt.yscale('log')
        plt.ylim([1e-4, 2])
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.savefig('mean_free_path.png')
        plt.show()


def main(argv):
    '''
    The Main prgram code
    Authors:
    '''
    del argv

    logging.info('Starting nuChic...')
    if not FLAGS.testing:
        nuchic = NuChic('run.yml')
        nuchic.run()

    logging.info('nuChic finished successfully!')


def nu_chic():
    """ Entry point for running the program """
    app.run(main)


if __name__ == "__main__":
    app.run(main)
