# =============================================================================
# 1. DÉFINITION DES CONFIGURATIONS (GRID SEARCH VAE)
# =============================================================================

CONFIGS = {
    # La Baseline (références issues des valeurs par défaut du script VAE)
    "baseline":     {"nl": 3, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5},
    
    # Variations de Layers (num_layer)
    "layers_2":     {"nl": 2, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5},
    "layers_4":     {"nl": 4, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.5},
    
    # Variations de Heads (num_head)
    "heads_2":      {"nl": 3, "nh": 2,  "f": 16,  "bs": 128, "cce": 0.5},
    "heads_8":      {"nl": 3, "nh": 8,  "f": 16,  "bs": 128, "cce": 0.5},

    # Variations de factor
    "factor_8":     {"nl": 3, "nh": 4,  "f": 8,  "bs": 128, "cce": 0.5},
    "factor_32":    {"nl": 3, "nh": 4,  "f": 32,  "bs": 128, "cce": 0.5},

    # Variations de Batch Size
    "batch_64":     {"nl": 3, "nh": 4,  "f": 16,  "bs": 64,  "cce": 0.5},
    "batch_256":    {"nl": 3, "nh": 4,  "f": 16,  "bs": 256, "cce": 0.5},
 
    # Variations de Categorical Cross Entropy (cce_weight)
    "cce_005":      {"nl": 3, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.25},
    "cce_02":       {"nl": 3, "nh": 4,  "f": 16,  "bs": 128, "cce": 0.75}
}

# =============================================================================
# 2. RÈGLES SNAKEMAKE
# =============================================================================

rule all:
    input:
        [
            f"checkpoints_VAE48_nl{c['nl']}_nh{c['nh']}_f{c['f']}_bs{c['bs']}_cce{c['cce']}/gen_data.npy"
            for c in CONFIGS.values()
        ]

rule train_vae:
    output:
        real = "checkpoints_VAE48_nl{nl}_nh{nh}_f{f}_bs{bs}_cce{cce}/real_data.npy",
        gen  = "checkpoints_VAE48_nl{nl}_nh{nh}_f{f}_bs{bs}_cce{cce}/gen_data.npy"
    log:
        "checkpoints_VAE48_nl{nl}_nh{nh}_f{f}_bs{bs}_cce{cce}/training.log"
    resources:
        gpu=1
    shell:
        """
        python -u AD4T_train_vae.py \
            --num_layer {wildcards.nl} \
            --num_head {wildcards.nh} \
            --factor {wildcards.f} \
            --batch_size {wildcards.bs} \
            --cce_weight {wildcards.cce} > {log} 2>&1
        """