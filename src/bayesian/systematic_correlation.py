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
