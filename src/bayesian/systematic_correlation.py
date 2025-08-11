#!/usr/bin/env python
'''
Systematic uncertainty correlation management for Bayesian inference

This module provides functionality to handle systematic uncertainty correlations based on 
user-defined correlation tags in configuration files.

MAIN CLASSES:
- SystematicInfo: Data structure for individual systematic uncertainty information
- SystematicCorrelationManager: Core class managing correlation groups and covariance calculation

KEY FEATURES:
- Parse correlation tags from config (e.g., 'jec:alice', 'taa:global', 'tracking:uncor')
- Build correlation groups based on user-defined tags (no assumptions about tag meanings)
- Create correlation-aware covariance matrices for likelihood calculations
- Validation and debugging tools for correlation structure

CORRELATION MODEL:
- Systematics with same correlation tag are fully correlated
- Systematics with different correlation tags are uncorrelated
- Special tag 'uncor' creates diagonal (uncorrelated) contributions
- Flexible: correlation tags can be experiment names, energy scales, or arbitrary user-defined groups

USAGE EXAMPLE:
    # Config file specifies correlation structure
    observable_list:
      - observable: 'jet_pt_alice'
        sys_data: ['jec:alice', 'taa:global']
      - observable: 'jet_pt_cms' 
        sys_data: ['jec:cms', 'taa:global']
    
    # Create and use correlation manager
    correlation_manager = SystematicCorrelationManager()
    correlation_manager.parse_configuration(parsed_observables)
    correlation_manager.register_observable_ranges(observable_ranges)
    
    # Calculate correlation-aware covariance matrix
    systematic_cov = correlation_manager.create_systematic_covariance_matrix(
        systematic_uncertainties, systematic_names, n_features
    )
authors: Jingyu Zhang (2025)
'''
import logging
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SystematicInfo:
    """
    Store information about a systematic uncertainty
    """
    base_name: str        # e.g., 'jec', 'taa'  
    correlation_tag: str  # e.g., 'group1', 'alice', 'experiment_a', etc. (user-defined)
    full_name: str       # e.g., 'jec:group1', 'taa:alice'
    is_uncorrelated: bool = False

    def __post_init__(self):
        self.is_uncorrelated = (self.correlation_tag.lower() == 'uncor')


class SystematicCorrelationManager:
    """
    Manages systematic uncertainty correlations based on user configuration.
    Makes no assumptions about the meaning of correlation tags.
    """
    
    def __init__(self):
        # Map correlation tags to lists of (observable, feature_range, systematic)
        self.correlation_groups: Dict[str, List[Tuple[str, int, int, str]]] = defaultdict(list)
        # Structure: correlation_tag -> [(observable_label, start_idx, end_idx, systematic_full_name), ...]
        
        # Map systematic full names to their info
        self.systematic_info: Dict[str, SystematicInfo] = {}
        # Structure: systematic_full_name -> SystematicInfo
        
        # Map observables to their expected systematics
        self.observable_systematics: Dict[str, List[str]] = {}
        # Structure: observable_label -> [systematic_full_names]
        
        # Store all unique systematic full names for consistent ordering
        self.all_systematic_names: List[str] = []

    def parse_configuration(self, parsed_observables: List[Tuple[str, List[str], List[str]]]):
        """
        Parse the configuration format with correlation tags.
        Makes no assumptions about tag meanings - they are just user-defined strings.
        
        :param parsed_observables: List of (observable_name, sys_data_list, sys_theory_list)
                                  where sys_data_list contains items like ['jec:group1', 'taa:alice']
        """
        logger.info("Parsing systematic correlation configuration...")
        
        all_systematic_full_names = set()
        
        for obs_name, sys_data_list, sys_theory_list in parsed_observables:
            self.observable_systematics[obs_name] = []
            
            # Process data systematics
            for sys_spec in sys_data_list:
                if ':' in sys_spec:
                    base_name, correlation_tag = sys_spec.split(':', 1)
                else:
                    # If no tag specified, treat as uncorrelated
                    base_name = sys_spec
                    correlation_tag = 'uncor'
                    logger.warning(f"No correlation tag for {sys_spec}, treating as uncorrelated")
                
                full_name = f"{base_name}:{correlation_tag}"
                
                # Store systematic info
                sys_info = SystematicInfo(
                    base_name=base_name,
                    correlation_tag=correlation_tag,
                    full_name=full_name
                )
                self.systematic_info[full_name] = sys_info
                self.observable_systematics[obs_name].append(full_name)
                all_systematic_full_names.add(full_name)
                
                logger.debug(f"  {obs_name}: {sys_spec} -> {full_name}")
        
        # Create consistent ordering of systematic names
        self.all_systematic_names = sorted(list(all_systematic_full_names))
        logger.info(f"Found {len(self.all_systematic_names)} unique systematics: {self.all_systematic_names}")

    def register_observable_ranges(self, observable_ranges: List[Tuple[int, int, str]]):
        """
        Register which features belong to which observables and build correlation groups.
        
        :param observable_ranges: List of (start_idx, end_idx, observable_label)
        """
        logger.info("Building correlation groups from observable ranges...")
        
        # Clear existing correlation groups
        self.correlation_groups.clear()
        
        # Build correlation groups by going through each observable
        for start_idx, end_idx, obs_label in observable_ranges:
            if obs_label in self.observable_systematics:
                for sys_full_name in self.observable_systematics[obs_label]:
                    sys_info = self.systematic_info[sys_full_name]
                    
                    if not sys_info.is_uncorrelated:
                        # Group by correlation tag (whatever the user specified)
                        correlation_tag = sys_info.correlation_tag
                        self.correlation_groups[correlation_tag].append(
                            (obs_label, start_idx, end_idx, sys_full_name)
                        )
        
        # Log correlation groups for debugging
        logger.info("Correlation groups built:")
        for group_tag, group_members in self.correlation_groups.items():
            logger.info(f"  Group '{group_tag}': {len(group_members)} entries")
            for obs_label, start, end, sys_name in group_members:
                logger.debug(f"    {sys_name} on {obs_label} (features {start}:{end})")

    def get_systematic_names_for_observable(self, observable_label: str) -> List[str]:
        """Get list of systematic full names for a given observable"""
        return self.observable_systematics.get(observable_label, [])

    def get_all_systematic_names(self) -> List[str]:
        """Get consistent ordering of all systematic names"""
        return self.all_systematic_names.copy()

    def create_systematic_covariance_matrix(self, 
                                          systematic_uncertainties: np.ndarray,
                                          systematic_names: List[str],
                                          n_features: int) -> np.ndarray:
        """
        Create systematic covariance matrix based on correlation structure.
        
        :param systematic_uncertainties: Array of shape (n_features, n_systematics)
        :param systematic_names: List of systematic full names (should match self.all_systematic_names)
        :param n_features: Total number of features
        :return: Covariance matrix of shape (n_features, n_features)
        """
        logger.debug("Creating systematic covariance matrix...")
        
        # Validate input
        if systematic_names != self.all_systematic_names:
            logger.warning("Systematic names don't match expected ordering")
        
        systematic_cov = np.zeros((n_features, n_features))
        
        # Process each systematic uncertainty
        for sys_idx, sys_full_name in enumerate(systematic_names):
            if sys_idx >= systematic_uncertainties.shape[1]:
                logger.warning(f"Systematic {sys_full_name} index {sys_idx} exceeds uncertainty array size")
                continue
                
            sys_uncertainty = systematic_uncertainties[:, sys_idx]
            sys_info = self.systematic_info.get(sys_full_name)
            
            if sys_info is None:
                logger.warning(f"No info found for {sys_full_name}, treating as uncorrelated")
                systematic_cov += np.diag(sys_uncertainty**2)
                continue
                
            if sys_info.is_uncorrelated:
                # Add as diagonal contribution
                systematic_cov += np.diag(sys_uncertainty**2)
                logger.debug(f"  {sys_full_name}: uncorrelated (diagonal)")
            else:
                # Find all features that should be correlated for this systematic
                correlation_tag = sys_info.correlation_tag
                correlated_feature_indices = []
                
                if correlation_tag in self.correlation_groups:
                    for obs_label, start_idx, end_idx, group_sys_name in self.correlation_groups[correlation_tag]:
                        # Check if this systematic applies to this observable
                        if group_sys_name == sys_full_name:
                            correlated_feature_indices.extend(range(start_idx, end_idx))
                
                if correlated_feature_indices:
                    # Remove duplicates and sort
                    correlated_feature_indices = sorted(set(correlated_feature_indices))
                    
                    # Create correlation block for these features
                    correlation_contribution = self._create_correlation_block(
                        sys_uncertainty, correlated_feature_indices, n_features
                    )
                    systematic_cov += correlation_contribution
                    
                    logger.debug(f"  {sys_full_name}: correlated across {len(correlated_feature_indices)} features")
                else:
                    # Fallback to diagonal if no correlation group found
                    systematic_cov += np.diag(sys_uncertainty**2)
                    logger.warning(f"  {sys_full_name}: no correlation group found, using diagonal")
        
        logger.debug(f"Systematic covariance matrix completed, Frobenius norm: {np.linalg.norm(systematic_cov):.2e}")
        return systematic_cov

    def _create_correlation_block(self, 
                                uncertainties: np.ndarray, 
                                correlated_indices: List[int], 
                                n_features: int) -> np.ndarray:
        """
        Create correlation block for a specific set of features.
        Assumes full correlation: C_ij = σ_i * σ_j for correlated features.
        
        :param uncertainties: Full uncertainty array (n_features,)
        :param correlated_indices: List of feature indices that should be correlated
        :param n_features: Total number of features
        :return: Covariance matrix with correlation block
        """
        cov_matrix = np.zeros((n_features, n_features))
        
        # Create fully correlated block: C_ij = σ_i * σ_j
        for i_idx in correlated_indices:
            for j_idx in correlated_indices:
                cov_matrix[i_idx, j_idx] = uncertainties[i_idx] * uncertainties[j_idx]
        
        return cov_matrix

    def get_correlation_summary(self) -> Dict:
        """
        Get summary information about the correlation structure for debugging/validation
        """
        summary = {
            'n_systematics': len(self.all_systematic_names),
            'n_observables': len(self.observable_systematics),
            'n_correlation_groups': len(self.correlation_groups),
            'correlation_groups': {},
            'uncorrelated_systematics': []
        }
        
        # Group information
        for group_tag, group_members in self.correlation_groups.items():
            summary['correlation_groups'][group_tag] = {
                'n_entries': len(group_members),
                'systematics': list(set([sys_name for _, _, _, sys_name in group_members])),
                'observables': list(set([obs_name for obs_name, _, _, _ in group_members]))
            }
        
        # Uncorrelated systematics
        for sys_full_name, sys_info in self.systematic_info.items():
            if sys_info.is_uncorrelated:
                summary['uncorrelated_systematics'].append(sys_full_name)
        
        return summary

    def validate_configuration(self) -> List[str]:
        """
        Validate the correlation configuration and return list of warnings/errors
        """
        warnings = []
        
        # Check for systematics that appear in config but no correlation groups
        for sys_full_name, sys_info in self.systematic_info.items():
            if not sys_info.is_uncorrelated:
                found_in_group = False
                for group_members in self.correlation_groups.values():
                    if any(sys_name == sys_full_name for _, _, _, sys_name in group_members):
                        found_in_group = True
                        break
                
                if not found_in_group:
                    warnings.append(f"Systematic {sys_full_name} has correlation tag but no correlation group")
        
        # Check for empty correlation groups
        for group_tag, group_members in self.correlation_groups.items():
            if len(group_members) <= 1:
                warnings.append(f"Correlation group '{group_tag}' has only {len(group_members)} member(s)")
        
        return warnings

    def to_dict(self) -> Dict:
        """
        Convert SystematicCorrelationManager to a serializable dictionary for HDF5 storage.
        
        :return: Dictionary representation of the correlation manager
        """
        return {
            'correlation_groups': dict(self.correlation_groups),  # Convert defaultdict to dict
            'systematic_info': {
                full_name: {
                    'base_name': info.base_name,
                    'correlation_tag': info.correlation_tag,
                    'full_name': info.full_name,
                    'is_uncorrelated': info.is_uncorrelated
                }
                for full_name, info in self.systematic_info.items()
            },
            'observable_systematics': dict(self.observable_systematics),
            'all_systematic_names': self.all_systematic_names,
            'class_name': 'SystematicCorrelationManager'  # For validation during loading
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SystematicCorrelationManager':
        """
        Reconstruct SystematicCorrelationManager from serialized dictionary.
        
        :param data: Dictionary representation from to_dict()
        :return: Reconstructed SystematicCorrelationManager instance
        """
        # Validate that this is the right type of data
        class_name = data.get('class_name')
        if isinstance(class_name, np.ndarray):
            class_name = str(class_name.item())  # Convert numpy scalar to string
        if class_name != 'SystematicCorrelationManager':
            raise ValueError(f"Invalid data format for SystematicCorrelationManager: {class_name}")
        
        # Create new instance
        manager = cls()
        
        # Restore correlation_groups (convert back to defaultdict)
        # Handle potential numpy arrays from HDF5
        manager.correlation_groups = defaultdict(list)
        for tag, group_list in data['correlation_groups'].items():
            # Ensure tag is a string
            tag_str = str(tag.item()) if isinstance(tag, np.ndarray) else str(tag)
            
            # Convert group_list items if they are numpy arrays
            processed_group_list = []
            for item in group_list:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    # Convert each element to proper type
                    obs_label = str(item[0].item()) if isinstance(item[0], np.ndarray) else str(item[0])
                    start_idx = int(item[1].item()) if isinstance(item[1], np.ndarray) else int(item[1])
                    end_idx = int(item[2].item()) if isinstance(item[2], np.ndarray) else int(item[2])
                    sys_name = str(item[3].item()) if isinstance(item[3], np.ndarray) else str(item[3])
                    processed_group_list.append((obs_label, start_idx, end_idx, sys_name))
                else:
                    processed_group_list.append(item)
            
            manager.correlation_groups[tag_str] = processed_group_list
        
        # Restore systematic_info with type conversion
        manager.systematic_info = {}
        for full_name, info_dict in data['systematic_info'].items():
            # Ensure all strings are proper strings, not numpy arrays
            full_name_str = str(full_name.item()) if isinstance(full_name, np.ndarray) else str(full_name)
            base_name = str(info_dict['base_name'].item()) if isinstance(info_dict['base_name'], np.ndarray) else str(info_dict['base_name'])
            correlation_tag = str(info_dict['correlation_tag'].item()) if isinstance(info_dict['correlation_tag'], np.ndarray) else str(info_dict['correlation_tag'])
            full_name_from_dict = str(info_dict['full_name'].item()) if isinstance(info_dict['full_name'], np.ndarray) else str(info_dict['full_name'])
            is_uncorrelated = bool(info_dict['is_uncorrelated'].item()) if isinstance(info_dict['is_uncorrelated'], np.ndarray) else bool(info_dict['is_uncorrelated'])
            
            manager.systematic_info[full_name_str] = SystematicInfo(
                base_name=base_name,
                correlation_tag=correlation_tag,
                full_name=full_name_from_dict,
                is_uncorrelated=is_uncorrelated
            )
        
        # Restore other attributes with type conversion
        manager.observable_systematics = {}
        for obs_label, sys_list in data['observable_systematics'].items():
            obs_label_str = str(obs_label.item()) if isinstance(obs_label, np.ndarray) else str(obs_label)
            sys_list_str = [str(item.item()) if isinstance(item, np.ndarray) else str(item) for item in sys_list]
            manager.observable_systematics[obs_label_str] = sys_list_str
        
        # Convert all_systematic_names to proper strings
        manager.all_systematic_names = [
            str(item.item()) if isinstance(item, np.ndarray) else str(item) 
            for item in data['all_systematic_names']
        ]
        
        return manager
