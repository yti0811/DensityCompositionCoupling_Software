# Density–Composition Coupling Software

Python scripts for *Density–composition coupling in compressible interfacial adsorption of a two-state soft solute*.

## Setup

Keep all numbered `.py` files in one folder. Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Scripts and commands

Run scripts **01–14 in order** for the main manuscript workflow.

| Script | Objective | Run |
| --- | --- | --- |
| `01_equilibrium_regime_map.py` | Calculate equilibrium states and classify regimes. | `python 01_equilibrium_regime_map.py` |
| `02_representative_linear_response.py` | Calculate linear response for representative states. | `python 02_representative_linear_response.py` |
| `03_global_linear_response.py` | Calculate linear response across the parameter survey. | `python 03_global_linear_response.py` |
| `04_soft_density_regime_certification.py` | Check density-regime classification. | `python 04_soft_density_regime_certification.py` |
| `05_assemble_manuscript_results.py` | Collect results for manuscript tables. | `python 05_assemble_manuscript_results.py` |
| `06_numerical_convergence_validation.py` | Check numerical convergence. | `python 06_numerical_convergence_validation.py` |
| `07_KKT_boundary_validation.py` | Check constrained-boundary solutions. | `python 07_KKT_boundary_validation.py` |
| `08_generate_manuscript_figures.py` | Generate manuscript figures. | `python 08_generate_manuscript_figures.py` |
| `09_export_source_data.py` | Export figure source data. | `python 09_export_source_data.py` |
| `10_finite_wall_crossover_refined.py` | Find free-volume crossover points at finite wall strength. | `python 10_finite_wall_crossover_refined.py` |
| `11_weak_wall_adsorption_sign_audit.py` | Check adsorption signs and convergence near crossover. | `python 11_weak_wall_adsorption_sign_audit.py` |
| `12_parameter_sensitivity_analysis.py` | Test sensitivity to wall strength, state energy, and bulk solute fraction. | `python 12_parameter_sensitivity_analysis.py` |
| `13_sensitivity_grid_convergence.py` | Check grid convergence of sensitivity results. | `python 13_sensitivity_grid_convergence.py` |
| `14_excess_grand_potential.py` | Compare excess grand potentials of Cases I and II. | `python 14_excess_grand_potential.py` |

## Notes

- Keep `01_equilibrium_regime_map.py` available: scripts 10–14 use its model functions. Run **10 before 11** and **12 before 13** because the later scripts read earlier CSV outputs.
- Scripts **10–14 produce CSV files only**. Scripts 10–13 run in quick mode by default. For the more thorough revision checks, add `--mode certified` (and `--convergence` for script 10); script 14 has no mode option.
- Quick-mode results are exploratory and should not be used as final numerical certification. Check the CSV status fields and numerical convergence before reporting results.
