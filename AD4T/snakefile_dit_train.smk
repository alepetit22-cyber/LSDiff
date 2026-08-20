# =============================================================================
# 1. DÉFINITION DES CONFIGURATIONS (GRID SEARCH DIT)
# =============================================================================

CONFIGS = {
    ## VAE 1 ##
    "baseline_VAE1":    {"nl": 2, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 160, "d": 7, "nh": 8},
    "1num_heads_4":     {"nl": 2, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 160, "d": 7, "nh": 4},
    
    # Variations de hidden size
    "1hidden_80":       {"nl": 2, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 80, "d": 7, "nh": 4},
    "1num_heads_2":     {"nl": 2, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 80, "d": 7, "nh": 2},
    
    "1hidden_320":      {"nl": 2, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 320, "d": 7, "nh": 8},
    "1num_heads_16":    {"nl": 2, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 320, "d": 7, "nh": 16},
    
    # Variations de depth
    "1depth_5":         {"nl": 2, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 160, "d": 5, "nh": 8},
    "1depth_9":         {"nl": 2, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 160, "d": 9, "nh": 8},


    ## VAE 2 ##
    "Baseline_VAE2":    {"nl": 3, "nh": 4,  "f": 16,  "bs": 64, "cce": 0.5,  "hd": 160, "d": 7, "nh": 8},
    "2num_heads_4":     {"nl": 3, "nh": 4,  "f": 16,  "bs": 64, "cce": 0.5,  "hd": 160, "d": 7, "nh": 4},
    
    # Variations de hidden size
    "2hidden_80":       {"nl": 3, "nh": 4,  "f": 16,  "bs": 64, "cce": 0.5,  "hd": 80, "d": 7, "nh": 4},
    "2num_heads_2":     {"nl": 3, "nh": 4,  "f": 16,  "bs": 64, "cce": 0.5,  "hd": 80, "d": 7, "nh": 2},
    
    "2hidden_220":      {"nl": 3, "nh": 4,  "f": 16,  "bs": 64, "cce": 0.5,  "hd": 320, "d": 7, "nh": 8},
    "2num_heads_16":    {"nl": 3, "nh": 4,  "f": 16,  "bs": 64, "cce": 0.5,  "hd": 320, "d": 7, "nh": 16},
    
    # Variations de depth
    "2depth_5":         {"nl": 3, "nh": 4,  "f": 16,  "bs": 64, "cce": 0.5,  "hd": 160, "d": 5, "nh": 8},
    "2depth_9":         {"nl": 3, "nh": 4,  "f": 16,  "bs": 64, "cce": 0.5,  "hd": 160, "d": 9, "nh": 8},


    ## VAE 3 ##
    "baseline_VAE3":    {"nl": 4, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 160, "d": 7, "nh": 8},
    "3num_heads_4":     {"nl": 4, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 160, "d": 7, "nh": 4},
    
    # Variations de hidden size
    "3hidden_80":       {"nl": 4, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 80, "d": 7, "nh": 4},
    "3num_heads_2":     {"nl": 4, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 80, "d": 7, "nh": 2},
    
    "3hidden_220":      {"nl": 4, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 320, "d": 7, "nh": 8},
    "3num_heads_16":    {"nl": 4, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 320, "d": 7, "nh": 16},
    
    # Variations de depth
    "3depth_5":         {"nl": 4, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 160, "d": 5, "nh": 8},
    "3depth_9":         {"nl": 4, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5,  "hd": 160, "d": 9, "nh": 8},
}

# =============================================================================
# 2. RÈGLES SNAKEMAKE
# =============================================================================

rule all:
    input:
        [
            f"checkpoints_pred16_hist32_VAE_nl{c['nl']}_nh{c['nh']}_f{c['f']}_bs{c['bs']}_cce{c['cce']}_DIT_hd{c['hd']}_d{c['d']}_nh{c['nh']}/real_data.npy"
            for c in CONFIGS.values()
        ]

rule train_dit:
    output:
        real = "checkpoints_pred16_hist32_VAE_nl{nl}_nh{nh}_f{f}_bs{bs}_cce{cce}_DIT_hd{hd}_d{d}_nh{nh}/real_data.npy",
        gen  = "checkpoints_pred16_hist32_VAE_nl{nl}_nh{nh}_f{f}_bs{bs}_cce{cce}_DIT_hd{hd}_d{d}_nh{nh}/gen_data.npy"
    log:
        "checkpoints_pred16_hist32_VAE_nl{nl}_nh{nh}_f{f}_bs{bs}_cce{cce}_DIT_hd{hd}_d{d}_nh{nh}/training.log"
    resources:
        gpu=1
    shell:
        """
        python3 -u AD4T_train_dit.py \
            --num_layer {wildcards.nl} \
            --num_head_VAE {wildcards.nh} \
            --factor {wildcards.f} \
            --batch_size {wildcards.bs} \
            --cce_weight {wildcards.cce} \
            --hidden_size {wildcards.hd} \
            --depth {wildcards.d} \
            --num_head_DIT {wildcards.nh} > {log} 2>&1
        """