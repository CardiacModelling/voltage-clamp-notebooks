# This branch:

Original experiments to see if tau_sum made any sense.


# Voltage clamp notebooks

## Intro

Recently as 92 not uncommon for patch clampers to build own amplifiers (SAKAKIBARA)
and perform detailed modelling (OTHER REF)
Commercialisation and specialisation: Experimenter, Hardware expert, Modeller seperate people
This paper and notebooks explain patch clamp to modellers
Code and equations provided for target audience modellers who analyse voltage clamp data, but hope figures and text also useful for experimenters

## Paper contents

1. [notebooks](How it works)
2. [experiments](How to calibrate it / examples of quality of fit)
    - I suspect that everyone can get slightly better results by refitting to their amp
    - Maaaaay redo Chon's experiments with the model cell for this (and redo mine with his cell)
3. Implications for fitting

## Journal

- eLife (living article: runnable stuff)
- https://www.sciencedirect.com/journal/the-journal-of-precision-medicine-health-and-disease

## Why another one

- Chon's first papers use it, Roman's paper uses it, ex293 paper uses it.
- Lei, Clark & all show its application to INa
- Lei simulator paper will be (should be!) aimed at massive market of non-programming experiments
- Some paper can still introduce or use the current clamp model
- INa artefact paper will use it behind the scenes
- **This** paper specifically for _people who analyse voltage clamp data with models_, and who want to _understand amplifier by modeling_ it.

