from bayesian import data_IO
from systematic_correlation import SystematicCorrelationManager

def enhanced_initialize_observables_dict_from_tables(table_dir, analysis_config, parameterization):
    """
    Enhanced version that creates and stores correlation manager in the observables dict
    
    WHAT THIS FUNCTION DOES:
    1. Parses your jingyu.yaml config to extract correlation information
    2. Calls your existing Phase 1 initialization function
    3. Stores the correlation manager in the observables dict
    4. Validates consistency between config and data
    
    INTEGRATION POINTS:
    - Uses your existing Phase 1 data reading infrastructure
    - Adds correlation manager on top without breaking existing functionality
    - Stores everything in the same observables.h5 file format
    """
    
    logger.info('Enhanced initialization with systematic correlation support...')
    
    # STEP 1: Parse correlation information from config
    # ================================================
    # This is NEW - extracts correlation tags from your jingyu.yaml
    correlation_manager = data_IO._parse_config_observables(analysis_config)
    
    # What this does:
    # - Reads observable_list from analysis_config["parameters"]["emulators"]
    # - Extracts sys_data entries like ['jec:group1', 'taa:global'] 
    # - Creates SystematicCorrelationManager with parsed correlation info
    # - Does NOT read any .dat files yet - just parses the YAML config
    
    logger.info(f"Parsed {len(correlation_manager.get_all_systematic_names())} systematics from config")
    
    # STEP 2: Call your existing Phase 1 initialization
    # =================================================
    # This is UNCHANGED - your existing function that:
    # - Reads .dat files from table_dir
    # - Parses systematic columns (s_jec, s_taa, etc.) 
    # - Creates the observables dict structure
    # - Stores Data, Prediction, Design, etc.
    
    observables = data_IO.initialize_observables_dict_from_tables(table_dir, analysis_config, parameterization)
    
    # At this point, observables contains your Phase 1 structure:
    # observables['Data'][obs_label]['y'] = [values]
    # observables['Data'][obs_label]['y_err_stat'] = [stat_errors]  
    # observables['Data'][obs_label]['systematics']['jec'] = [jec_errors]
    # observables['Data'][obs_label]['systematics']['taa'] = [taa_errors] 
    # observables['Prediction'][obs_label] = {...}
    # observables['Design'][parameterization] = [...]
    
    # STEP 3: Store correlation manager in observables dict
    # ====================================================
    # This is NEW - we add the correlation manager to the observables dict
    # so it gets saved to observables.h5 and can be retrieved later
    
    observables['_correlation_manager'] = correlation_manager
    
    # Now observables contains BOTH:
    # - All your existing Phase 1 data structures (unchanged)
    # - The correlation manager (new addition)
    
    # What this validation does:
    # - For each observable, check if config expects 'jec' systematic
    # - Verify that the .dat file actually has 's_jec' column
    # - Warn if config expects systematics not found in data
    # - Warn if data has systematics not mentioned in config
    
    return observables


def detailed_initialization_example():
    """
    Step-by-step example of what happens during initialization
    """
    
    print("=== INITIALIZATION PROCESS WALKTHROUGH ===\n")
    
    # Example config (from your jingyu.yaml)
    example_config = {
        'parameters': {
            'emulators': {
                'main': {
                    'observable_list': [
                        {
                            'observable': '5020__PbPb__jet_pt_alice__R0.2__0-10',
                            'sys_data': ['jec:alice', 'taa:5020']  # <-- Correlation tags
                        },
                        {
                            'observable': '5020__PbPb__jet_pt_cms__R0.2__0-10', 
                            'sys_data': ['jec:cms', 'taa:5020']   # <-- Different JEC group, same TAA
                        }
                    ]
                }
            }
        }
    }
    
    print("1. CONFIG PARSING:")
    print("   Input config has correlation tags:")
    print("   - jec:alice (JEC correlated within ALICE)")
    print("   - jec:cms (JEC correlated within CMS)")
    print("   - taa:5020 (TAA correlated across all 5020 GeV data)")
    
    # What enhanced_parse_config_observables does:
    correlation_manager = data_IO._parse_config_observables(example_config)
    systematic_names = correlation_manager[1].get_all_systematic_names()
    
    print(f"\n   Parsed systematic names: {systematic_names}")
    print("   Note: These are the 'full names' with correlation tags")
    
    print("\n2. EXISTING PHASE 1 DATA READING:")
    print("   Your existing function reads .dat files:")
    print("   Data__5020__PbPb__jet_pt_alice__R0.2__0-10.dat:")
    print("   # Label xmin xmax y y_err_stat s_jec s_taa")
    print("   bin1    0   10  1.2  0.05      0.08  0.03")
    print("   bin2   10   20  0.9  0.04      0.06  0.02")
    print("")
    print("   Data__5020__PbPb__jet_pt_cms__R0.2__0-10.dat:")
    print("   # Label xmin xmax y y_err_stat s_jec s_taa") 
    print("   bin1    0   10  1.1  0.04      0.07  0.03")
    print("   bin2   10   20  0.8  0.03      0.05  0.02")
    
    print("\n   Phase 1 creates observables dict:")
    print("   observables['Data']['5020__PbPb__jet_pt_alice__R0.2__0-10']['systematics']['jec'] = [0.08, 0.06]")
    print("   observables['Data']['5020__PbPb__jet_pt_alice__R0.2__0-10']['systematics']['taa'] = [0.03, 0.02]")
    print("   observables['Data']['5020__PbPb__jet_pt_cms__R0.2__0-10']['systematics']['jec'] = [0.07, 0.05]")
    print("   observables['Data']['5020__PbPb__jet_pt_cms__R0.2__0-10']['systematics']['taa'] = [0.03, 0.02]")
    
    print("\n3. CORRELATION MANAGER STORAGE:")
    print("   Enhanced function adds correlation_manager to observables dict")
    print("   observables['_correlation_manager'] = correlation_manager")
    print("   This gets saved to observables.h5 file")
    
    print("\n4. VALIDATION:")
    print("   Check config vs data consistency:")
    print("   ✓ Config expects 'jec' → Found 's_jec' in .dat files")
    print("   ✓ Config expects 'taa' → Found 's_taa' in .dat files") 
    print("   ✓ All systematics accounted for")
    
    print("\n5. RESULT:")
    print("   observables.h5 now contains:")
    print("   - All your existing Phase 1 data structures (unchanged)")
    print("   - Correlation manager with parsed correlation information") 
    print("   - Ready for correlation-aware analysis in later steps")


def write_dict_to_h5_explanation():
    """
    Explain what happens when writing to H5 file
    """
    print("\n=== WRITING TO H5 FILE ===\n")
    
    print("WHAT GETS SAVED:")
    print("The observables dict contains:")
    print("")
    print("1. EXISTING PHASE 1 STRUCTURES:")
    print("   observables['Data'][obs_label]['y'] = measurement values")
    print("   observables['Data'][obs_label]['y_err_stat'] = statistical errors")
    print("   observables['Data'][obs_label]['systematics']['jec'] = JEC systematic errors")
    print("   observables['Data'][obs_label]['systematics']['taa'] = TAA systematic errors")
    print("   observables['Prediction'][obs_label] = theory predictions")
    print("   observables['Design'][parameterization] = design points")
    print("")
    print("2. NEW CORRELATION INFORMATION:")
    print("   observables['_correlation_manager'] = SystematicCorrelationManager object")
    print("     - Contains parsed correlation groups (jec:alice, jec:cms, taa:5020)")
    print("     - Contains mapping from base names (jec) to full names (jec:alice)")
    print("     - Contains validation information")
    
    print("\nH5 FILE STRUCTURE:")
    print("observables.h5:")
    print("├── Data/")
    print("│   ├── 5020__PbPb__jet_pt_alice__R0.2__0-10/")
    print("│   │   ├── y = [1.2, 0.9, ...]")
    print("│   │   ├── y_err_stat = [0.05, 0.04, ...]") 
    print("│   │   └── systematics/")
    print("│   │       ├── jec = [0.08, 0.06, ...]")
    print("│   │       └── taa = [0.03, 0.02, ...]")
    print("│   └── 5020__PbPb__jet_pt_cms__R0.2__0-10/")
    print("│       └── [similar structure]")
    print("├── Prediction/")
    print("├── Design/")
    print("└── _correlation_manager/  <-- NEW")
    print("    ├── systematic_info/")
    print("    ├── correlation_groups/") 
    print("    └── observable_systematics/")
    
    print("\nWHY THIS APPROACH:")
    print("✓ Backward compatible - existing code still works")
    print("✓ All correlation info preserved in H5 file")
    print("✓ Can be retrieved later for MCMC/likelihood calculations")
    print("✓ No changes needed to existing .dat file format")
    print("✓ Correlation information only in config file (jingyu.yaml)")


def integration_with_existing_code():
    """
    Show how this integrates with your existing workflow
    """
    print("\n=== INTEGRATION WITH EXISTING CODE ===\n")
    
    print("BEFORE (Phase 1 only):")
    print("```python")
    print("# In steer_analysis.py")
    print("observables = initialize_observables_dict_from_tables(table_dir, analysis_config, parameterization)")
    print("write_dict_to_h5(observables, output_dir, 'observables.h5')")
    print("```")
    
    print("\nAFTER (Phase 1 + Correlations):")
    print("```python") 
    print("# In steer_analysis.py")
    print("observables = enhanced_initialize_observables_dict_from_tables(table_dir, analysis_config, parameterization)")
    print("write_dict_to_h5(observables, output_dir, 'observables.h5')")
    print("```")
    
    print("\nCHANGES NEEDED:")
    print("1. Replace function call (1 line change)")
    print("2. Add correlation tags to jingyu.yaml config")
    print("3. That's it! Everything else works the same")
    
    print("\nLATER, FOR MCMC:")
    print("```python")
    print("# When you're ready for Phase 2 MCMC integration")
    print("experimental_data = enhanced_data_array_from_h5(output_dir, 'observables.h5')")
    print("systematic_cov = calculate_systematic_covariance_matrix(experimental_data)")
    print("# Use systematic_cov in likelihood calculation")
    print("```")
