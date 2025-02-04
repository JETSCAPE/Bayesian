'''
Main script to steer Bayesian inference studies for heavy-ion jet analysis

authors: J.Mulligan, R.Ehlers
Based in part on JETSCAPE/STAT code.
'''

import argparse
import logging
import os
import shutil
import yaml
from pathlib import Path

from bayesian import data_IO, preprocess_input_data, mcmc
from bayesian import plot_input_data, plot_emulation, plot_mcmc, plot_qhat, plot_closure, plot_analyses

from bayesian import common_base, helpers
from bayesian.emulation import base

logger = logging.getLogger(__name__)


####################################################################################################################
class SteerAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', **kwargs):

        # Initialize config file
        self.config_file = config_file
        self.initialize()

        logger.info(self)

    #---------------------------------------------------------------
    # Initialize config
    #---------------------------------------------------------------
    def initialize(self):
        logger.info('Initializing class objects')

        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.output_dir = config['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Data inputs
        self.observable_table_dir = config['observable_table_dir']
        self.observable_config_dir = config['observable_config_dir']

        # Configure which functions to run
        self.initialize_observables = config['initialize_observables']
        self.preprocess_input_data = config['preprocess_input_data']
        self.fit_emulators = config['fit_emulators']
        self.run_mcmc = config['run_mcmc']
        self.run_closure_tests = config['run_closure_tests']
        self.plot = config['plot']

        # Configuration of different analyses
        self.analyses = config['analyses']

    #---------------------------------------------------------------
    # Main function
    #---------------------------------------------------------------
    def run_analysis(self):
        # Add logging to file
        _root_log = logging.getLogger()
        _root_log.addHandler(logging.FileHandler(os.path.join(self.output_dir, 'steer_analysis.log'), 'w'))

        # Also write analysis config to shared directory
        shutil.copy(self.config_file, Path(self.output_dir) / "steer_analysis_config.yaml")

        # Loop through each analysis
        with helpers.progress_bar() as progress:
            analysis_task = progress.add_task("[deep_sky_blue1]Running analysis...", total=len(self.analyses))

            for analysis_name, analysis_config in self.analyses.items():

                # Loop through the parameterizations
                parameterization_task = progress.add_task("[deep_sky_blue2]parameterization", total=len(analysis_config['parameterizations']))
                for parameterization in analysis_config['parameterizations']:

                    # Initialize design points, predictions, data, and uncertainties
                    # We store them in a dict and write/read it to HDF5
                    if self.initialize_observables:
                        # Just indicate that it's working
                        initialization_task = progress.add_task("[deep_sky_blue4]Initializing...", total=None)
                        progress.start_task(initialization_task)
                        logger.info("")
                        logger.info('========================================================================')
                        logger.info(f'Initializing model: {analysis_name} ({parameterization} parameterization)...')
                        observables = data_IO.initialize_observables_dict_from_tables(self.observable_table_dir,
                                                                                    analysis_config,
                                                                                    parameterization)
                        data_IO.write_dict_to_h5(observables,
                                                os.path.join(self.output_dir, f'{analysis_name}_{parameterization}'),
                                                filename='observables.h5')
                        progress.update(initialization_task, advance=100, visible=False)

                    if self.preprocess_input_data:
                        # Just indicate that it's working
                        preprocess_task = progress.add_task("[deep_sky_blue4]Preprocessing...", total=None)
                        progress.start_task(preprocess_task)
                        logger.info("")
                        logger.info('------------------------------------------------------------------------')
                        logger.info(f'Preprocessing input data: {analysis_name} ({parameterization} parameterization)...')

                        preprocessing_config = preprocess_input_data.PreprocessingConfig(
                            analysis_name=analysis_name,
                            parameterization=parameterization,
                            analysis_config=analysis_config,
                            config_file=self.config_file,
                        )
                        # NOTE: Strictly speaking, we don't want the emulation config here. However,
                        #       We often need the observable filter, and it doesn't cost anything to
                        #       construct here, so we just go for it.
                        #emulation_config = emulation.EmulationConfig.from_config_file(
                        #    analysis_name=analysis_name,
                        #    parameterization=parameterization,
                        #    analysis_config=analysis_config,
                        #    config_file=self.config_file,
                        #)
                        observables_smoothed = preprocess_input_data.preprocess(
                            preprocessing_config=preprocessing_config,
                        )
                        data_IO.write_dict_to_h5(observables_smoothed,
                                                os.path.join(self.output_dir, f'{analysis_name}_{parameterization}'),
                                                filename='observables_preprocessed.h5')
                        progress.update(preprocess_task, advance=100, visible=False)

                    # Fit emulators and write them to file
                    if self.fit_emulators:
                        # Just indicate that it's working
                        emulation_task = progress.add_task("[deep_sky_blue4]Emulating...", total=None)
                        progress.start_task(emulation_task)
                        logger.info('------------------------------------------------------------------------')
                        logger.info(f'Fitting emulators for {analysis_name}_{parameterization}...')
                        emulation_config = base.EmulationConfig.from_config_file(
                            analysis_name=analysis_name,
                            parameterization=parameterization,
                            analysis_config=analysis_config,
                            config_file=self.config_file,
                        )
                        base.fit_emulators(emulation_config)
                        progress.update(emulation_task, advance=100, visible=False)

                    # Run MCMC
                    if self.run_mcmc:
                        # Just indicate that it's working
                        mcmc_task = progress.add_task("[deep_sky_blue4]Running MCMC...", total=None)
                        progress.start_task(mcmc_task)
                        logger.info("")
                        logger.info('------------------------------------------------------------------------')
                        logger.info(f'Running MCMC for {analysis_name}_{parameterization}...')
                        mcmc_config = mcmc.MCMCConfig(analysis_name=analysis_name,
                                                      parameterization=parameterization,
                                                      analysis_config=analysis_config,
                                                      config_file=self.config_file)
                        mcmc.run_mcmc(mcmc_config)
                        progress.update(mcmc_task, advance=100, visible=False)

                    # Run closure tests -- one for each validation design point
                    #   - Use validation point as pseudodata
                    #   - Use emulator already trained on training points
                    if self.run_closure_tests:
                        n_design_points = analysis_config['validation_indices'][1] - analysis_config['validation_indices'][0]
                        closure_test_task = progress.add_task("[deep_sky_blue4]Running closure tests...", total=n_design_points)
                        progress.start_task(closure_test_task)
                        logger.info("")
                        logger.info('------------------------------------------------------------------------')
                        for design_point_index in range(n_design_points):
                            logger.info(f'Running closure tests for {analysis_name}_{parameterization}, validation_index={design_point_index}...')
                            mcmc_config = mcmc.MCMCConfig(analysis_name=analysis_name,
                                                          parameterization=parameterization,
                                                          analysis_config=analysis_config,
                                                          config_file=self.config_file,
                                                          closure_index=design_point_index)
                            mcmc.run_mcmc(mcmc_config, closure_index=design_point_index)
                            progress.update(closure_test_task, advance=1)
                        progress.update(closure_test_task, visible=False)

                    progress.update(parameterization_task, advance=1)
                # Hide once we're done!
                progress.update(parameterization_task, visible=False)

                progress.update(analysis_task, advance=1)

        # Plots for individual analysis
        for analysis_name,analysis_config in self.analyses.items():
            for parameterization in analysis_config['parameterizations']:

                if any(self.plot.values()):
                    logger.info('========================================================================')
                    logger.info(f'Plotting for {analysis_name} ({parameterization} parameterization)...')
                    logger.info("")

                if self.plot["input_data"]:
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting input data for {analysis_name}_{parameterization}...')
                    emulation_config = base.EmulationConfig.from_config_file(
                        analysis_name=analysis_name,
                        parameterization=parameterization,
                        analysis_config=analysis_config,
                        config_file=self.config_file,
                    )
                    plot_input_data.plot(emulation_config)
                    logger.info(f'Done!')
                    logger.info("")

                if self.plot['emulators']:

                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting emulators for {analysis_name}_{parameterization}...')
                    emulation_config = base.EmulationConfig.from_config_file(
                        analysis_name=analysis_name,
                        parameterization=parameterization,
                        analysis_config=analysis_config,
                        config_file=self.config_file,
                    )
                    plot_emulation.plot(emulation_config)
                    logger.info(f'Done!')
                    logger.info("")

                if self.plot['mcmc']:
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting MCMC for {analysis_name}_{parameterization}...')
                    mcmc_config = mcmc.MCMCConfig(analysis_name=analysis_name,
                                                  parameterization=parameterization,
                                                  analysis_config=analysis_config,
                                                  config_file=self.config_file)
                    plot_mcmc.plot(mcmc_config)
                    logger.info(f'Done!')
                    logger.info("")

                if self.plot['qhat']:
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting qhat results {analysis_name}_{parameterization}...')
                    mcmc_config = mcmc.MCMCConfig(analysis_name=analysis_name,
                                                  parameterization=parameterization,
                                                  analysis_config=analysis_config,
                                                  config_file=self.config_file)
                    plot_qhat.plot(mcmc_config)
                    logger.info(f'Done!')
                    logger.info("")

                if self.plot['closure_tests']:
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting closure test results {analysis_name}_{parameterization}...')
                    mcmc_config = mcmc.MCMCConfig(analysis_name=analysis_name,
                                                  parameterization=parameterization,
                                                  analysis_config=analysis_config,
                                                  config_file=self.config_file)
                    plot_closure.plot(mcmc_config)
                    logger.info(f'Done!')
                    logger.info("")

        # Plots across multiple analyses
        if self.plot['across_analyses']:
            # NOTE: This is a departure from the standard API, but we need a convention for how
            #       to pass multiple analyses, so we'll just go with it for now.
            plot_analyses.plot(self.analyses, self.config_file, self.output_dir)


####################################################################################################################
if __name__ == '__main__':
    helpers.setup_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Jet Bayesian Analysis')
    parser.add_argument('-c', '--configFile',
                        help='Path of config file for analysis',
                        action='store', type=str,
                        default='../config/jet_substructure.yaml', )
    args = parser.parse_args()

    logger.info('Configuring...')
    logger.info(f'  configFile: {args.configFile}')

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        msg = f'File {args.configFile} does not exist! Exiting!'
        logger.info(msg)
        raise ValueError(msg)

    analysis = SteerAnalysis(config_file=args.configFile)
    analysis.run_analysis()
