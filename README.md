# MXINT-Based Exponent-Sign Approximation for Self-Attention Pruning in Vision and Diffusion Transformers

## Overview
Propose MXINT8 shared exponent and element sign approximated Q, K for lightweight Q*K used for attention pruning
- reduce compute power and memory transfer in approximated Q*K stage

## Details
#### Target workloads
- Vision transformer: DeiT-tiny, DeiT-small, DeiT-base
- Diffusion transformer: DiT-XL/2 (256x256), PixArt- $\alpha$ (256x256)

This repo contains:
1. Running MXINT dynamic quantization
2. Running approximated top-k attention pruning
   - Proposed: MXINT8 shared exponent and element sign
   - Sanger (related work): MXINT4
   - EXION (related work): Two-step leading one
   - ELSA (related work): Sign orthogonal projection    
4. Evaluation (accuracy for **deit**, FID for **DiT, PixArt - $\alpha$**)
