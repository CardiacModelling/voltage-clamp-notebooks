# Filter questions

Simulations and experiments to work out details of filtering and its interaction with series resistance compensation in the artefact model.

Initial simulations in questions.ipynb

Updated simulations and experimental data in answers.ipynb

All performed on a HEKA EPC-10 on the model cell [MC10](https://www.heka.com/products/products_main.html#physiol_mc10), in whole-cell model except where indicated otherwise.

## Experiment notes

Protocol: Disable filter2, highest acquisition setting, save I and V
- Hold at \-100
- 100ms at \-100, 100ms at \+100, 
- 100ms at \-30, 100ms at \+30
- 100ms at \-40, 100ms at \+20

Zoom in on region t=99.9 to t=101.5
Only this transition is intended to be used.
The other steps are included as back-ups with (A) a smaller magnitude and (B) no symmetry around 0mV.

### Experiment 1 (no-comp-tau)

- No filter2, filter1 auto, highest sample rate
- Disable prediction, Compensation low but not off (1%)
- Repeat protocol with all 4 r\_series\_tau settings
- Gain 1mV/pA
- E408

### Experiment 2a (stim filter)

- No filter2, filter1 30kHz, highest sample rate, no comp, no pred
- Repeat with both stimulus filter settings
- E409 (repeated with better gain)

### Experiment 2b (filter1)

- No filter2, stim filter default, highest sample rate, no comp, no pred
- Repeat with all filter1 settings
- E410

### Experiment 2c (filter2)

- Filter1 30kHz, stim filter default, highest sample rate, no comp, no pred
- Repeat with filter2 at 10, 5, 2, 1 kHz
- E411 (keeps changing filter1, so will have to double check)

### Experiment 2d (cfast)

- No filter2, filter1 30kHz, stim filter default, highest sample rate, no comp, no pred
- Repeat with Cfast x-0.4, x-0.2, x, x+0.2, x+0.4
- E412 (with loads of extras)
- E416 (Cell-attached, with extras, but HQ30 filter)
- E417 (Cell-attached, with extras, using Bessel 100kHz)
- E418 (Cell-attached, with filter1 changes, incl Cm comp not needed)

### Experiment 2e (cslow)

- No filter2, filter1 30kHz, stim filter default, highest sample rate, no comp, no pred
- Repeat with Cslow x-2, x-1, x, x+1, x+2
-  E413 (wth extras)

### Experiment 3a

- No filter2, filter1 by software, stim filter default, highest sample rate
- Comp 70%, Pred 0%
- Repeat with all r\_series\_tau settings
- E414 (with lower settings than 70 as well)

### Experiment 3b

- No filter2, filter1 by software, stim filter default, highest sample rate
- Comp 70%, Pred 70%
- Repeat with all r\_series\_tau settings
- E415 (with extras)
