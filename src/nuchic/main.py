""" Main driver code. """

from tqdm import tqdm
import argparse

from .utilities import logger
from ._nuchic import InteractionLoader, InteractionFactory

from . import run_modes
from .config import settings
from ._version import version


def splash(output):
    """ Print a splash screen on startup """
    output("""
+=================================================================+
|                                                                 | 
|       d8b   db db    db  .o88b. db   db d888888b  .o88b.        |
|       888o  88 88    88 d8P  Y8 88   88   `88'   d8P  Y8        |
|       88V8o 88 88    88 8P      88ooo88    88    8P             |
|       88 V8o88 88    88 8b      88~~~88    88    8b             |
|       88  V888 88b  d88 Y8b  d8 88   88   .88.   Y8b  d8        |
|       VP   V8P ~Y8888P'  `Y88P' YP   YP Y888888P  `Y88P'        |
|                                                                 |
+-----------------------------------------------------------------+
|                                                                 |
|    Version: {:52}|
|    Authors: Joshua Isaacson, William Jay, Alessandro Lovato,    | 
|             Pedro A. Machado, Noemi Rocco                       | 
|                                                                 |
+=================================================================+
""".format(version))


class NuChic:
    """ Main driver class for the nuchic code. """
    def __init__(self, run_card):
        settings().load(run_card)
        self.nevents = settings().nevents
        self.calc = run_modes.RunMode(settings().get_run('mode', 'generate'))

    def run(self):
        """ Generate events for the given input settings. """
        events = []
        for _ in tqdm(range(self.nevents), ncols=80):
            event = self.calc.generate_one_event()
            events.append(event)

        return self.calc.finalize(events)


def nuchic():
    """ External entry point. """
    splash(print)

    # Arguments for the nuchic code
    description = 'Nuchic: The modular neutrino event generator.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-m', '--model_fsi', action='store_const',
                        const=True, default=False,
                        help='List the available interaction models for fsi')
    parser.add_argument('-i', '--input', default='run.yml',
                        help='Input file to be used')

    args = parser.parse_args()

    # Startup running of main code
    logger.init('nuchic', 'nuchic.log')
    splash(logger.debug)
    InteractionLoader.load_plugins(["./src/Plugins/lib"])
    if args.model_fsi:
        InteractionFactory.list()
        return
    driver = NuChic(args.input)
    driver.run()


if __name__ == '__main__':
    nuchic()
