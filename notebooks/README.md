# Modelling patch-clamp experiments

When analysing data from whole-cell patch-clamp experiments, it can be useful to have a model of both the biological system of interest _and_ the experimental set up.
In these notebooks we retrace the steps taken in the supplement to [Lei et al., 2020](https://doi.org/10.1098/rsta.2019.0348), and construct (1) a model of a patch-clamp experiment with various experimental artefacts, and (2) a model of the corrections applied by patch-clamp amplifiers to mitigate these effects.
The exposition draws heavily on a book chapter by [Sigworth (1995a)](https://doi.org/10.1007/978-1-4419-1229-9_4).

I have tried to keep things as to-the-point as possible, but a lot of extra detail is provided in the appendices.

## Modelling patch-clamp experiments [![github](../img/github.svg)](artefacts-1-modelling-patch-clamp.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/artefacts-1-modelling-patch-clamp.ipynb)

The first notebook describes the uncompensated patch-clamp set up, and shows how to derive both an electrical schematic and an ODE model.

## Modelling electronic compensation [![github](../img/github.svg)](artefacts-2-compensation.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/artefacts-2-compensation.ipynb)

In the second notebook we update the model to include the compensation circuitry commonly used in patch-clamp amplifiers.

## Simulating a manual patch clamp experiment [![github](../img/github.svg)](artefacts-3-simulations.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/artefacts-3-simulations.ipynb)

(Unfinished) In this notebook, we walk through the steps of a manual patch-clamp experiment.

## Simplified models [![github](../img/github.svg)](artefacts-4-simplified.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/artefacts-4-simplified.ipynb)

(Unfinished) In notebook number five we derive simplified models of the compensated voltage clamp setup, which can be used in fitting.

## Summary [![github](../img/github.svg)](artefacts-6-summary.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/artefacts-6-summary.ipynb)

(Unfinished) Finally, we present two simplified models (with currents in pA and currents in A/F), in equation & Myokit form.

## Appendices

- Electronics
  - A1. Ideal op amps [![github](../img/github.svg)](appendix-A1-op-amp.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-A1-op-amp.ipynb)
  - A2. Laplace transforms and filters [![github](../img/github.svg)](appendix-A2-laplace-and-filters.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-A2-laplace-and-filters.ipynb)
  - A3. Non-ideal op amps [![github](../img/github.svg)](appendix-A3-non-ideal-op-amp.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-A3-non-ideal-op-amp.ipynb)
  - A4. Bessel low-pass filters [![github](../img/github.svg)](appendix-A4-bessel-filters.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-A4-bessel-filters.ipynb)
  - A5. Bessel filter ODEs [![github](../img/github.svg)](appendix-A5-bessel-filter-odes.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-A5-bessel-filter-odes.ipynb)
- Extended models
  - B1. Models without compensation [![github](../img/github.svg)](appendix-B1-uncompensated-models.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-B1-uncompensated-models.ipynb)
  - B2. Sigworth 1983/1995 Rs compensation [![github](../img/github.svg)](appendix-B2-sigworth-rs.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-B2-sigworth-rs.ipynb)
- Parameter names and values
  - C1. Names & symbols  [![github](../img/github.svg)](appendix-C1-symbols.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-C1-symbols.ipynb)
  - C2. Default parameter values used in examples  [![github](../img/github.svg)](appendix-C2-parameter-defaults.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-C2-parameter-defaults.ipynb)
  - C3. Parameter values, estimates for different amplifiers etc. [![github](../img/github.svg)](appendix-C3-parameter-values.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-C3-parameter-values.ipynb)
- Remaining noise and errors
  - D1. Strategies for dealing with experimental error [![github](../img/github.svg)](appendix-D1-strategies.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-D1-strategies.ipynb)
  - D2. Stochastic and periodic noise [![github](../img/github.svg)](appendix-D2-inspecting-noise.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-D2-inspecting-noise.ipynb)
  - D3. Liquid junction potential [![github](../img/github.svg)](appendix-D3-liquid-junction-potential.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-D3-liquid-junction-potential.ipynb)
  - D4. Leak (unfinished) [![github](../img/github.svg)](appendix-D4-leak.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-D4-leak.ipynb)
  - D5. Handling remaining capacitance artefacts (unfinished) [![github](../img/github.svg)](appendix-D5-remaining-Cp-artefacts.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-D5-remaining-Cp-artefacts.ipynb)
- Estimating Rs and Cm
  - E1. Estimating Rs and Cm; a one-shot approach [![github](../img/github.svg)](appendix-E1-rs-cm-one-shot.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-E1-rs-cm-one-shot.ipynb)
  - E2. Estimating Rs and Cm; an iterative approach (unfinished) [![github](../img/github.svg)](appendix-E2-rs-cm-iterative.ipynb) [![nbviewer](../img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/voltage-clamp-notebooks/tree/main/notebooks/appendix-E2-rs-cm-iterative.ipynb)
