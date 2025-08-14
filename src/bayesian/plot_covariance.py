#!/usr/bin/env python
"""
Covariance and correlation matrix visualization for Bayesian inference

This module provides visualization tools for understanding the covariance structure
in the Bayesian analysis, including:

1. Statistical uncertainty covariance (diagonal)
2. Individual systematic uncertainty group covariances 
3. Emulator uncertainty covariance
4. Total combined covariance matrix

Key features:
- Heatmap visualizations of covariance/correlation matrices
- Component-wise breakdown of uncertainty sources
- Diagnostic plots for validation

Authors: Jingyu Zhang (2025)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm, SymLogNorm

from bayesian import data_IO, mcmc
from bayesian.systematic_correlation import SystematicCorrelationManager

logger = logging.getLogger(__name__)

# Set matplotlib style
sns.set_context('paper', rc={'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})


def plot_all_covariance_components(
    config: mcmc.MCMCConfig,
    output_dir: str,
    parameter_point: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (20, 16),
    cmap: str = 'RdBu_r',
    save_matrices: bool = True
) -> None:
    """
    Create comprehensive visualization of all covariance components.
    
    :param config: MCMC configuration object
    :param output_dir: Directory to save plots
    :param parameter_point: Parameter point for emulator evaluation (if None, use MAP)
    :param figsize: Figure size for the combined plot
    :param cmap: Colormap for heatmaps
    :param save_matrices: Whether to save matrices to disk for debugging
    """
    
    logger.info("Creating comprehensive covariance component visualization...")
    
    # Create plot directory
    plot_dir = Path(output_dir) / 'covariance_plots'
    plot_dir.mkdir(exist_ok=True)
    
    # Load experimental data and setup
    experimental_data = data_IO.data_array_from_h5(
        config.output_dir, 
        config.observables_filename
    )
    
    # Get emulator predictions at specified parameter point
    if parameter_point is None:
        # Use MAP estimate if available
        try:
            results = data_IO.read_dict_from_h5(config.output_dir, config.mcmc_outputfilename, verbose=False)
            chain = results['chain']
            n_walkers, n_steps, n_params = chain.shape
            posterior = chain.reshape((n_walkers*n_steps, n_params))
            parameter_point = mcmc.map_parameters(posterior)
            logger.info(f"Using MAP parameter point: {parameter_point}")
        except Exception as e:
            logger.warning(f"Could not load MAP parameters: {e}, using parameter center")
            # Use center of parameter space as fallback
            try:
                # Get parameter bounds from analysis config
                param_config = config.analysis_config['parameterization'][config.parameterization]
                param_min = np.array(param_config['min'])
                param_max = np.array(param_config['max'])
                parameter_point = (param_min + param_max) / 2
                logger.info(f"Using parameter center: {parameter_point}")
            except Exception as e2:
                logger.warning(f"Could not determine parameter center: {e2}, using zeros")
                parameter_point = np.zeros(len(config.analysis_config['parameterization'][config.parameterization]['names']))
    
    emulator_predictions = _get_emulator_predictions(config, parameter_point.reshape(1, -1))
    
    # Extract covariance components
    cov_components = _extract_covariance_components(experimental_data, emulator_predictions)
    
    # Create main comparison plot
    _plot_covariance_comparison(cov_components, plot_dir, figsize, cmap)
    
    # Create individual detailed plots
    _plot_individual_covariance_matrices(cov_components, plot_dir, cmap)
    
    # Create systematic group correlation plots
    _plot_systematic_group_correlations(cov_components, plot_dir, cmap)
    
    # Create correlation matrix plots
    _plot_correlation_matrices(cov_components, plot_dir, cmap)
    
    # Create diagnostic plots
    _plot_covariance_diagnostics(cov_components, plot_dir)
    
    # Save matrices for debugging if requested
    if save_matrices:
        _save_covariance_matrices(cov_components, plot_dir)
    
    logger.info(f"Covariance plots saved to {plot_dir}")


def _extract_covariance_components(
    experimental_data: Dict,
    emulator_predictions: Dict
) -> Dict[str, np.ndarray]:
    """
    Extract and compute all covariance matrix components.
    
    :param experimental_data: Experimental data dictionary from data_array_from_h5
    :param emulator_predictions: Emulator predictions dictionary
    :return: Dictionary of covariance matrices
    """
    
    n_features = len(experimental_data['y'])
    cov_components = {}
    
    # 1. Statistical uncertainty covariance (diagonal)
    stat_errors = np.array(experimental_data['y_err_stat'])
    cov_components['statistical'] = np.diag(stat_errors**2)
    logger.info(f"Statistical covariance: {cov_components['statistical'].shape}, diagonal with {np.sum(stat_errors**2):.2e} total variance")
    
    # 2. Systematic uncertainty covariances by group
    if 'correlation_manager' in experimental_data:
        correlation_manager = experimental_data['correlation_manager']
        systematic_uncertainties = experimental_data['y_err_syst']
        systematic_names = experimental_data['systematic_names']
        
        # Total systematic covariance
        cov_components['systematic_total'] = correlation_manager.create_systematic_covariance_matrix(
            systematic_uncertainties, systematic_names, n_features
        )
        
        # Individual systematic group covariances
        cov_components.update(_compute_systematic_group_covariances(
            correlation_manager, systematic_uncertainties, systematic_names, n_features
        ))
        
        logger.info(f"Systematic total covariance: {cov_components['systematic_total'].shape}, "
                   f"total variance {np.trace(cov_components['systematic_total']):.2e}")
    else:
        logger.warning("No correlation manager found - creating zero systematic covariance")
        cov_components['systematic_total'] = np.zeros((n_features, n_features))
    
    # 3. Emulator uncertainty covariance
    if emulator_predictions and 'cov' in emulator_predictions:
        # Take the first sample's covariance (since we evaluated at single point)
        cov_components['emulator'] = emulator_predictions['cov'][0]
        logger.info(f"Emulator covariance: {cov_components['emulator'].shape}, "
                   f"total variance {np.trace(cov_components['emulator']):.2e}")
    else:
        logger.warning("No emulator covariance available - creating zero matrix")
        cov_components['emulator'] = np.zeros((n_features, n_features))
    
    # 4. Total combined covariance
    cov_components['total'] = (
        cov_components['statistical'] + 
        cov_components['systematic_total'] + 
        cov_components['emulator']
    )
    
    logger.info(f"Total covariance: {cov_components['total'].shape}, "
               f"total variance {np.trace(cov_components['total']):.2e}")
    
    return cov_components


def _compute_systematic_group_covariances(
    correlation_manager: SystematicCorrelationManager,
    systematic_uncertainties: np.ndarray,
    systematic_names: List[str],
    n_features: int
) -> Dict[str, np.ndarray]:
    """
    Compute covariance matrix for each systematic uncertainty group separately.
    """
    
    group_covariances = {}
    
    # Get correlation groups from the manager
    correlation_groups = correlation_manager.correlation_groups
    
    for group_tag, group_members in correlation_groups.items():
        # Create covariance matrix for this group only
        group_cov = np.zeros((n_features, n_features))
        
        # Find which systematics belong to this group
        group_systematics = set()
        for obs_label, start_idx, end_idx, sys_full_name in group_members:
            group_systematics.add(sys_full_name)
        
        # Add contribution from each systematic in this group
        for sys_idx, sys_name in enumerate(systematic_names):
            if sys_name in group_systematics:
                sys_uncertainty = systematic_uncertainties[:, sys_idx]
                
                # Find correlated feature indices for this systematic
                correlated_indices = []
                for obs_label, start_idx, end_idx, group_sys_name in group_members:
                    if group_sys_name == sys_name:
                        correlated_indices.extend(range(start_idx, end_idx))
                
                if correlated_indices:
                    correlated_indices = sorted(set(correlated_indices))
                    
                    # Create correlation submatrix
                    for i in correlated_indices:
                        for j in correlated_indices:
                            if i < n_features and j < n_features:
                                group_cov[i, j] += sys_uncertainty[i] * sys_uncertainty[j]
        
        group_covariances[f'systematic_group_{group_tag}'] = group_cov
        logger.info(f"Systematic group '{group_tag}': variance {np.trace(group_cov):.2e}")
    
    # Also add uncorrelated systematics as a separate component
    uncorr_cov = np.zeros((n_features, n_features))
    for sys_idx, sys_name in enumerate(systematic_names):
        sys_info = correlation_manager.systematic_info.get(sys_name)
        if sys_info and sys_info.is_uncorrelated:
            sys_uncertainty = systematic_uncertainties[:, sys_idx]
            uncorr_cov += np.diag(sys_uncertainty**2)
    
    if np.trace(uncorr_cov) > 0:
        group_covariances['systematic_group_uncorrelated'] = uncorr_cov
        logger.info(f"Uncorrelated systematics: variance {np.trace(uncorr_cov):.2e}")
    
    return group_covariances


def _plot_covariance_comparison(
    cov_components: Dict[str, np.ndarray],
    plot_dir: Path,
    figsize: Tuple[int, int],
    cmap: str
) -> None:
    """
    Create a comparison plot showing all covariance components side by side.
    """
    
    # Select main components for comparison
    main_components = ['statistical', 'systematic_total', 'emulator', 'total']
    available_components = [comp for comp in main_components if comp in cov_components]
    
    n_components = len(available_components)
    fig, axes = plt.subplots(2, n_components, figsize=figsize)
    
    if n_components == 1:
        axes = axes.reshape(2, 1)
    
    for i, comp_name in enumerate(available_components):
        cov_matrix = cov_components[comp_name]
        
        # Plot covariance matrix (top row)
        ax_cov = axes[0, i]
        _plot_single_matrix(cov_matrix, ax_cov, f'{comp_name.replace("_", " ").title()} Covariance', 
                           cmap, matrix_type='covariance')
        
        # Plot correlation matrix (bottom row)  
        ax_corr = axes[1, i]
        corr_matrix = _covariance_to_correlation(cov_matrix)
        _plot_single_matrix(corr_matrix, ax_corr, f'{comp_name.replace("_", " ").title()} Correlation',
                           'RdBu_r', matrix_type='correlation')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'covariance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(plot_dir / 'covariance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created covariance comparison plot")


def _plot_individual_covariance_matrices(
    cov_components: Dict[str, np.ndarray],
    plot_dir: Path,
    cmap: str
) -> None:
    """
    Create individual detailed plots for each covariance matrix.
    """
    
    for comp_name, cov_matrix in cov_components.items():
        if comp_name.startswith('systematic_group_'):
            # Skip individual systematic groups for now (too many plots)
            continue
            
        fig, (ax_cov, ax_corr) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Covariance matrix
        _plot_single_matrix(cov_matrix, ax_cov, f'{comp_name.replace("_", " ").title()} Covariance',
                           cmap, matrix_type='covariance')
        
        # Correlation matrix
        corr_matrix = _covariance_to_correlation(cov_matrix)
        _plot_single_matrix(corr_matrix, ax_corr, f'{comp_name.replace("_", " ").title()} Correlation',
                           'RdBu_r', matrix_type='correlation')
        
        plt.tight_layout()
        filename = f'covariance_{comp_name}'
        plt.savefig(plot_dir / f'{filename}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plot_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("Created individual covariance matrix plots")


def _plot_systematic_group_correlations(
    cov_components: Dict[str, np.ndarray],
    plot_dir: Path,
    cmap: str = 'RdBu_r'
) -> None:
    """
    Create focused plots for systematic uncertainty group correlations.
    """
    
    # Find all systematic group components
    systematic_groups = []
    for comp_name in cov_components.keys():
        if comp_name.startswith('systematic_group_') and comp_name != 'systematic_group_uncorrelated':
            # Extract group name (e.g., 'cms', 'alice', '5020')
            group_name = comp_name.replace('systematic_group_', '')
            systematic_groups.append((group_name, comp_name))
    
    if not systematic_groups:
        logger.info("No systematic correlation groups found to plot")
        return
    
    # Create individual plots for each systematic group
    for group_name, comp_name in systematic_groups:
        cov_matrix = cov_components[comp_name]
        
        # Skip if matrix is all zeros
        if np.allclose(cov_matrix, 0):
            logger.info(f"Skipping systematic group '{group_name}' - all zeros")
            continue
        
        corr_matrix = _covariance_to_correlation(cov_matrix)
        
        fig, (ax_cov, ax_corr) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Covariance matrix
        _plot_single_matrix(cov_matrix, ax_cov, f'Systematic Group "{group_name}" Covariance',
                           cmap, matrix_type='covariance')
        
        # Correlation matrix with improved visualization
        _plot_single_matrix(corr_matrix, ax_corr, f'Systematic Group "{group_name}" Correlation',
                           'RdBu_r', matrix_type='correlation')
        
        plt.tight_layout()
        filename = f'systematic_group_{group_name}'
        plt.savefig(plot_dir / f'{filename}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plot_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created systematic group correlation plot for '{group_name}'")
    
    # Create a comparison plot if we have multiple groups
    if len(systematic_groups) > 1:
        n_groups = len(systematic_groups)
        fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 5))
        
        if n_groups == 1:
            axes = [axes]
        
        for i, (group_name, comp_name) in enumerate(systematic_groups):
            cov_matrix = cov_components[comp_name]
            if np.allclose(cov_matrix, 0):
                continue
                
            corr_matrix = _covariance_to_correlation(cov_matrix)
            _plot_single_matrix(corr_matrix, axes[i], f'Group "{group_name}"',
                               'RdBu_r', matrix_type='correlation')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'systematic_groups_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plot_dir / 'systematic_groups_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created systematic groups comparison plot")


def _plot_correlation_matrices(
    cov_components: Dict[str, np.ndarray],
    plot_dir: Path,
    cmap: str
) -> None:
    """
    Create a focused plot comparing main correlation matrices.
    """
    
    # Select components with interesting correlation structure
    correlation_components = []
    for comp_name in ['systematic_total', 'emulator', 'total']:
        if comp_name in cov_components:
            correlation_components.append(comp_name)
    
    if not correlation_components:
        return
    
    n_components = len(correlation_components)
    fig, axes = plt.subplots(1, n_components, figsize=(6*n_components, 5))
    
    if n_components == 1:
        axes = [axes]
    
    for i, comp_name in enumerate(correlation_components):
        cov_matrix = cov_components[comp_name]
        corr_matrix = _covariance_to_correlation(cov_matrix)
        
        _plot_single_matrix(corr_matrix, axes[i], 
                           f'{comp_name.replace("_", " ").title()}',
                           'RdBu_r', matrix_type='correlation')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'correlation_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(plot_dir / 'correlation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created correlation matrix comparison plot")


def _plot_single_matrix(
    matrix: np.ndarray,
    ax: plt.Axes,
    title: str,
    cmap: str,
    matrix_type: str = 'covariance'
) -> None:
    """
    Plot a single covariance or correlation matrix with appropriate scaling and colorbar.
    """
    
    if matrix_type == 'correlation':
        # Correlation matrices: use symmetric log norm for better visualization
        im = ax.imshow(
            matrix,
            norm=SymLogNorm(linthresh=1e-1, linscale=1, vmin=-1, vmax=1),
            cmap="RdBu_r",  # good for positive/negative
            aspect='equal', 
            origin='lower'
        )
        cbar_label = 'Correlation'
    else:
        # Covariance matrices: use log scale for better visibility
        matrix_abs = np.abs(matrix)
        matrix_abs[matrix_abs == 0] = np.nan  # Handle zeros
        
        if np.nanmax(matrix_abs) > 0:
            vmin = np.nanmin(matrix_abs[matrix_abs > 0])
            vmax = np.nanmax(matrix_abs)
            
            # Use symmetric log norm if matrix has both positive and negative values
            if np.min(matrix) < 0:
                linthresh = vmin * 10  # Linear threshold
                norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
            else:
                norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
        
        im = ax.imshow(matrix, cmap=cmap, aspect='equal', origin='lower', norm=norm)
        cbar_label = 'Covariance'
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label)
    
    # Formatting - remove feature index labels and grids
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Remove axis labels and ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')


def _plot_covariance_diagnostics(
    cov_components: Dict[str, np.ndarray],
    plot_dir: Path
) -> None:
    """
    Create diagnostic plots for covariance matrices.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Eigenvalue spectrum
    ax = axes[0, 0]
    for comp_name, cov_matrix in cov_components.items():
        if comp_name.startswith('systematic_group_'):
            continue
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
        eigenvals = eigenvals[eigenvals > 0]  # Remove numerical zeros
        
        if len(eigenvals) > 0:
            ax.semilogy(eigenvals, 'o-', label=comp_name, alpha=0.7)
    
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Condition number
    ax = axes[0, 1]
    condition_numbers = {}
    for comp_name, cov_matrix in cov_components.items():
        if comp_name.startswith('systematic_group_'):
            continue
        try:
            cond_num = np.linalg.cond(cov_matrix)
            condition_numbers[comp_name] = cond_num
        except:
            condition_numbers[comp_name] = np.inf
    
    names = list(condition_numbers.keys())
    values = list(condition_numbers.values())
    finite_values = [v if np.isfinite(v) else 1e16 for v in values]
    
    bars = ax.bar(names, finite_values)
    ax.set_yscale('log')
    ax.set_ylabel('Condition Number')
    ax.set_title('Matrix Condition Numbers')
    ax.tick_params(axis='x', rotation=45)
    
    # Color badly conditioned matrices
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val > 1e12:
            bar.set_color('red')
        elif val > 1e8:
            bar.set_color('orange')
    
    # 3. Diagonal elements
    ax = axes[1, 0]
    for comp_name, cov_matrix in cov_components.items():
        if comp_name.startswith('systematic_group_'):
            continue
        diagonal = np.diag(cov_matrix)
        diagonal = diagonal[diagonal > 0]
        
        if len(diagonal) > 0:
            ax.semilogy(diagonal, 'o-', label=comp_name, alpha=0.7)
    
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Diagonal Element (Variance)')
    ax.set_title('Diagonal Elements')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Trace contribution
    ax = axes[1, 1]
    traces = {}
    for comp_name, cov_matrix in cov_components.items():
        if comp_name != 'total':  # Skip total to see components
            traces[comp_name] = np.trace(cov_matrix)
    
    # Create pie chart of trace contributions
    if traces:
        positive_traces = {k: v for k, v in traces.items() if v > 0}
        if positive_traces:
            ax.pie(positive_traces.values(), labels=positive_traces.keys(), autopct='%1.1f%%')
            ax.set_title('Trace (Total Variance) Contribution')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'covariance_diagnostics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(plot_dir / 'covariance_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created covariance diagnostic plots")


def _covariance_to_correlation(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix.
    """
    diagonal = np.diag(cov_matrix).copy()  # Make a copy to avoid read-only issues
    diagonal[diagonal <= 0] = 1  # Avoid division by zero
    std_devs = np.sqrt(diagonal)
    
    # Correlation matrix: C_ij = Cov_ij / (std_i * std_j)
    correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
    
    # Ensure diagonal is exactly 1
    np.fill_diagonal(correlation_matrix, 1.0)
    
    return correlation_matrix


def _get_emulator_predictions(config: mcmc.MCMCConfig, parameter_point: np.ndarray) -> Dict:
    """Get emulator predictions using the SAME path as MCMC."""
    try:
        from bayesian.emulation import base
        
        print("DEBUG: Starting emulator predictions")
        print(f"DEBUG: Parameter point shape: {parameter_point.shape}")
        print(f"DEBUG: Parameter point values: {parameter_point}")
        
        # Step 1: Try to create emulation config
        print("DEBUG: Creating emulation config...")
        emulation_config = base.EmulatorOrganizationConfig.from_config_file(
            analysis_name=config.analysis_name,
            parameterization=config.parameterization,
            config_file=config.config_file,
            analysis_config=config.analysis_config
        )
        print(f"DEBUG: Emulation config created with {len(emulation_config.emulation_groups_config)} groups")
        print(f"DEBUG: Groups: {list(emulation_config.emulation_groups_config.keys())}")
        
        # Step 2: Try to load emulation results
        print("DEBUG: Loading emulation results...")
        emulation_results = emulation_config.read_all_emulator_groups()
        print(f"DEBUG: Loaded emulation results for: {list(emulation_results.keys())}")
        
        if not emulation_results:
            print("DEBUG: ERROR - No emulation results loaded!")
            return {}
        
        # Step 3: Try prediction
        print("DEBUG: Making prediction...")
        if parameter_point.ndim == 1:
            parameter_point = parameter_point.reshape(1, -1)
        print(f"DEBUG: Reshaped parameter point: {parameter_point.shape}")
        
        predictions = base.predict(
            parameter_point, 
            emulation_config,
            emulation_group_results=emulation_results
        )
        
        print(f"DEBUG: Predictions type: {type(predictions)}")
        print(f"DEBUG: Predictions keys: {predictions.keys() if predictions else 'None/Empty'}")
        
        if predictions and 'cov' in predictions:
            print(f"DEBUG: Covariance shape: {predictions['cov'].shape}")
            print(f"DEBUG: Covariance max: {np.max(np.abs(predictions['cov'])):.6e}")
            print(f"DEBUG: Covariance trace: {np.trace(predictions['cov'][0]):.6e}")
        else:
            print("DEBUG: ERROR - No covariance in predictions!")
            print(f"DEBUG: Available keys: {list(predictions.keys()) if predictions else 'No predictions'}")
        
        return predictions
        
    except Exception as e:
        print(f"DEBUG: EXCEPTION in emulator predictions: {e}")
        import traceback
        print(f"DEBUG: TRACEBACK: {traceback.format_exc()}")
        return {}

def _save_covariance_matrices(cov_components: Dict[str, np.ndarray], plot_dir: Path) -> None:
    """
    Save covariance matrices to files for debugging.
    """
    matrices_dir = plot_dir / 'matrices'
    matrices_dir.mkdir(exist_ok=True)
    
    for comp_name, matrix in cov_components.items():
        filename = matrices_dir / f'{comp_name}_covariance.txt'
        np.savetxt(filename, matrix, fmt='%.6e')
        
        # Also save correlation matrix
        corr_matrix = _covariance_to_correlation(matrix)
        corr_filename = matrices_dir / f'{comp_name}_correlation.txt'
        np.savetxt(corr_filename, corr_matrix, fmt='%.6f')
    
    logger.info(f"Saved covariance matrices to {matrices_dir}")


def plot(
    analysis_name: str,
    parameterization: str,
    analysis_config: Dict[str, Any],
    config_file: str
) -> None:
    """
    Main plotting function for covariance analysis following steer_analysis.py convention.
    
    :param analysis_name: Name of analysis (e.g., 'inclusive_jet')
    :param parameterization: Parameterization name (e.g., 'linear_in_log', 'quadratic')
    :param analysis_config: Analysis configuration dictionary
    :param config_file: Path to main config file
    """
    
    logger.info(f'Plotting covariance matrices for {analysis_name} ({parameterization} parameterization)...')
    
    # Create MCMC config object following the standard pattern
    mcmc_config = mcmc.MCMCConfig(
        analysis_name=analysis_name,
        parameterization=parameterization,
        analysis_config=analysis_config,
        config_file=config_file
    )
    
    # Run the plotting
    plot_all_covariance_components(
        config=mcmc_config,
        output_dir=mcmc_config.output_dir,
        parameter_point=None,  # Use MAP
        figsize=(20, 16),
        cmap='RdBu_r',
        save_matrices=True
    )
    
    logger.info('Done!')


# Convenience function for direct usage (keeps backward compatibility)
def plot_covariance_analysis(config: mcmc.MCMCConfig, output_dir: str) -> None:
    """
    Convenience function to create all covariance plots with sensible defaults.
    
    :param config: MCMC configuration object
    :param output_dir: Output directory for plots
    """
    plot_all_covariance_components(
        config=config,
        output_dir=output_dir,
        parameter_point=None,  # Use MAP
        figsize=(20, 16),
        cmap='RdBu_r',
        save_matrices=True
    )
