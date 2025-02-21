#!/usr/bin/env python3
#
# Fit to several experiments at once.
# Probably only runs on linux (in parallel)
#
# Definitely: Rs, Cm, Cf, g
# Maybe: tau_amp, Cf, Rf
#
import argparse
import os
import sys

import myokit
import pints
import numpy as np
import matplotlib.pyplot as plt

from methods import plots


# Check command line arguments
parser = argparse.ArgumentParser(description='Fit the artefact model')
parser.add_argument(
    '-c', '--config', type=int, default=1, choices=list(range(1, 17)))
parser.add_argument('--debug', action='store_true')
parser.add_argument('--redraw', action='store_true')
args = parser.parse_args()

config = args.config
debug = args.debug
redraw = args.redraw

# Defaults
root = f'figures-{config}'
if debug:
    root = 'figures-debug'
tofit = ['amp.Rs', 'amp.Cm', 'amp.Cp', 'amp.g']
tol = 1e-9
model_x0 = False

# Model 1: Stim filter, F1, Rs-lag, No summing speed
if config == 1:
    model_file = 'm-new-1.mmt'
elif config == 2:
    model_file = 'm-new-1.mmt'
    tofit += ['amp.tau_amp']
elif config == 3:
    model_file = 'm-new-1.mmt'
    tofit += ['amp.Cf']
elif config == 4:
    model_file = 'm-new-1.mmt'
    tofit += ['amp.tau_amp', 'amp.Cf']

# Model 3: Like model 1 but with a two-part Cfast
elif config == 5:
    model_file = 'm-new-3.mmt'
elif config == 6:
    model_file = 'm-new-3.mmt'
    tofit += ['amp.tau_amp']
elif config == 7:
    model_file = 'm-new-3.mmt'
    tofit += ['amp.Cf']
elif config == 8:
    model_file = 'm-new-3.mmt'
    tofit += ['amp.tau_amp', 'amp.Cf']

# Model 4: Like model 1 but with a summing speed added back in
elif config == 9:
    model_file = 'm-new-4.mmt'
elif config == 10:
    model_file = 'm-new-4.mmt'
    tofit += ['amp.tau_amp']
elif config == 11:
    model_file = 'm-new-4.mmt'
    tofit += ['amp.tau_sum']
elif config == 12:
    model_file = 'm-new-4.mmt'
    tofit += ['amp.Cf']
elif config == 13:
    model_file = 'm-new-4.mmt'
    tofit += ['amp.tau_amp', 'amp.tau_sum']
elif config == 14:
    model_file = 'm-new-4.mmt'
    tofit += ['amp.tau_amp', 'amp.Cf']
elif config == 15:
    model_file = 'm-new-4.mmt'
    tofit += ['amp.tau_sum', 'amp.Cf']
elif config == 16:
    model_file = 'm-new-4.mmt'
    tofit += ['amp.tau_amp', 'amp.tau_sum', 'amp.Cf']


data = {
    # 1. Rs tau varied but alpha 1%
    408: (0, 1, 2, 3),
    # 2a. Stim filter
    #409: (2, 3),
    # 2b. Filter 1
    410: (0, 1, 2),
    # 2c. Filter 2
    #411: (0, 1, 2, 3, 4),
    # 2d. C-fast
    412: (6, 2, 1, 5, 3, 4, 8),
    # 2d. C-fast on cell-attached
    #417: (2, 1, 0, 3, 4),
    # 2e. C-slow
    413: (6, 2, 1, 0, 3, 4, 7),
    # 3a. Rs-comp, varied tau (4x70%, 4x40%, 4x20%)
    #414: (4, 3, 1, 0, 5, 6, 7, 8, 9, 10, 11, 12),
    414: (3, 1, 0, 5, 6, 7, 8, 9, 10, 11, 12),
    # 3b. Rs-comp and pred, varied tau
    #415: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    #      19),
}

print('Fitting ' + ', '.join(tofit))
print(f'Storing in {root}')


# Create boundaries
p_bounds = {
    'amp.Rs': [0.004, 0.006],
    'amp.Cm': [17, 23],
    'amp.Cp': [5, 7],
    'amp.g': [1.8, 2.2],
    'amp.tau_amp': [1e-9, 1e-4],
    'amp.tau_sum': [1e-9, 1e-2],
    'amp.Cf': [1e-4, 1],
}
boundaries = pints.RectangularBoundaries(
    [p_bounds[var][0] for var in tofit], [p_bounds[var][1] for var in tofit])


# Create initial point
p_starts = {
    'amp.Rs': 0.005,
    'amp.Cm': 21,
    'amp.Cp': 6,
    'amp.g': 2.05,
    'amp.tau_amp': 1e-5,
    'amp.tau_sum': 1e-5,
    'amp.Cf': 5e-3,
}
if not model_x0:
    x0 = [p_starts[var] for var in tofit]


class Model(pints.ForwardModel):
    def __init__(self, path):
        m = myokit.load_model(path)
        m.check_units(myokit.UNIT_STRICT)
        self._dual_cfast = m.has_variable('amp.Cp_tau2')

        # Pre-pace at 0mV
        p = myokit.Protocol()
        p.add_step(level=0, duration=100)
        s = myokit.Simulation(m, p)
        s.set_tolerance(tol, tol)
        s.pre(99)

        # Create protocol
        p = myokit.Protocol()
        p.add_step(duration=100, level=-100)
        p.add_step(duration=100, level=+100)
        p.add_step(duration=100, level=-30)
        p.add_step(duration=100, level=+30)
        p.add_step(duration=100, level=-40)
        p.add_step(duration=100, level=+20)
        s.set_protocol(p)
        self._p = p

        # Parameters
        self._par = list(tofit)
        self._val = [m.get(k).eval() for k in self._par]

        # Store objects
        self._sim = s

    def dual_cfast(self):
        return self._dual_cfast

    def n_parameters(self):
        return len(self._par)

    def parameter_names(self):
        return list(self._par)

    def parameter_values(self):
        return np.array(self._val)

    def protocol(self):
        return self._p  # Sim has already cloned it

    def set_amplifier_settings_by_log(self, log):
        """ Update the amplifier settings according to the given log. """
        s = self._set_var

        # Fast capacitance
        if self._dual_cfast:
            s('amp.Cp_est1', log, 'c_fast_amp1_pF')
            s('amp.Cp_est2', log, 'c_fast_amp2_pF')
            s('amp.Cp_tau2', log, 'c_fast_tau2_us', 1e-3)
        else:
            s('amp.Cp_est', log, 'c_fast_pF')

        # Slow capacitance
        if log.meta['c_slow_compensation_enabled'] == 'true':
            s('amp.Cm_est', log, 'c_slow_pF')
        else:
            print('C slow compensation disabled')
            self._sim.set_constant('amp.Cm_est', 1e-6)

        # Series resistance
        s('amp.Rs_est', log, 'r_series_MOhm', 1e-3)
        if log.meta['r_series_compensation_enabled'] == 'true':
            s('amp.alpha', log, 'r_series_compensation_percent', 1e-2)
            s('amp.tau_rc', log, 'r_series_compensation_tau_us', 1e-3)
        else:
            self._sim.set_constant('amp.alpha', 0)
            self._sim.set_constant('amp.tau_rc', 10e-3)
        self._sim.set_constant('amp.beta', 0)

        # Filter 1
        s('amp.f1', log, 'filter1_kHz')

    def _set_var(self, var, log, value, factor=1):
        """ Change a simulation variable based on the meta data in a log. """
        if debug:
            print(f'Setting {var} to {float(log.meta[value]) * factor}')
        self._sim.set_constant(var, float(log.meta[value]) * factor)

    def simulate(self, parameters, times):
        """ Simulate: assuming amplifier settings already updated. """
        self._sim.reset()
        tmax = times[-1] + 0.005
        for k, v in zip(self._par, parameters):
            self._sim.set_constant(k, v)
        d = self._sim.run(tmax, log=['amp.I_obs'], log_times=times).npview()
        return d['amp.I_obs']


class DataLogProblem(pints.SingleOutputProblem):
    """
    Problem based on a data log. Reads it and updates the model when
    evaluating.
    """
    def __init__(self, model, experiment, series):
        self._path = f'data/E-{experiment}-{series:0>3}-FilterTest.zip'
        self._pfig = f'{root}/C-{config}-E-{experiment}-{series:0>3}'
        self._log = myokit.DataLog.load(self._path).npview()
        super().__init__(model, self._log.time(), self._log['0.Imon'])

    def evaluate(self, parameters):
        self._model.set_amplifier_settings_by_log(self._log)
        return super().evaluate(parameters)

    def debug_figure(self, parameters):
        fig = self._figure(parameters)
        plt.show()
        sys.exit(1)

    def figure(self, parameters, org_parameters=None, total_error_value=None,
                individual_error=None):
        print(f'Storing figure to {self._pfig}')
        fig = self._figure(parameters, org_parameters, total_error_value,
                           individual_error)
        fig.savefig(f'{self._pfig}.png')
        plt.close(fig)

    def _figure(self, parameters, org_parameters=None, total_error_value=None,
                individual_error=None):

        label = 'Simulation'
        values = self.evaluate(parameters)
        if org_parameters is not None:
            label = 'Optimised'
            org_values = self.evaluate(org_parameters)

        fig = plt.figure(figsize=(9, 7))
        fig.subplots_adjust(0.09, 0.07, 0.80, 0.99, hspace=0.35)
        grid = fig.add_gridspec(3, 1)

        ax0 = fig.add_subplot(grid[0, 0])
        ax0.set_xlabel('Time (ms)')
        ax0.set_ylabel('I (pA)')
        plots.protocol_bands(ax0, self._model.protocol())

        ax1 = fig.add_subplot(grid[1:, 0])
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('I (pA)')
        ax1.axhline(-212, lw=1, color='#ccc')
        ax1.axhline(+212, lw=1, color='#ccc')
        ax1.set_xlim(99.95, 101)

        for ax in (ax0, ax1):
            ax.plot(self._times, self._values, label='Experiment')
            if org_parameters is not None:
                ax.plot(self._times, org_values, color='#bbb',
                        label='Initial', zorder=0)
            ax.plot(self._times, values, label=label)

        # Show amplifier settings
        headers = ['Settings']
        panel = []
        panels = [panel]
        m = self._log.meta
        if self._model.dual_cfast():
            panel.extend([
                f'Cp_amp1: {m["c_fast_amp1_pF"]:.4} pF',
                f'Cp_amp2: {m["c_fast_amp2_pF"]:.4} pF',
                f'Cp_tau2: {m["c_fast_tau2_us"]:.4} us',
            ])
        else:
            panel.append(f'Cp_est: {m["c_fast_pF"]:.4} pF')
        panel.extend([
            f'Cm_est: {m["c_slow_pF"]:.4} pF',
            f'Cm enabled: {m["c_slow_compensation_enabled"]}',
            f'Filter1: {m["filter1"]}',
            f'Rs_est: {m["r_series_MOhm"]:.4} MOhm',
        ])
        if m['r_series_compensation_enabled'] == 'true':
            panel.extend([
                f'alpha: {m["r_series_compensation_percent"]} %',
                f'Rs_tau: {m["r_series_compensation_tau_us"]:.4} us',
            ])
        else:
            panel.append('Rs_comp disabled')

        # Parameters
        headers.append('Parameters')
        panels.append([f'{name.split(".")[1]}: {value:.4}' for name, value in
                       zip(self._model.parameter_names(), parameters)])

        # Show errors
        panel = []
        if total_error_value is not None:
            panel.append(f'Total: {total_error_value:.1f}')
            if individual_error is not None:
                e = individual_error(parameters)
                panel.append(f'Absolute: {e:.1f}')
                panel.append(f'Relative: {100 * e / total_error_value:.3} %')
        if len(panel):
            headers.append('Errors')
            panels.append(panel)
        plots.text_panels(fig, panels, headers, x=0.81, y=0.64)

        # Draw error (mean-squared only)
        if isinstance(individual_error, pints.MeanSquaredError):
            e = np.cumsum((self._values - values)**2)
            e /= e[-1]
            for ax in [ax0, ax1]:
                y = ax.get_ylim()
                r = y[0] + e * (y[1] - y[0])
                z = np.ones(r.shape) * y[0]
                ax.plot(self._times, r, 'tab:red', lw=1, alpha=0.1, zorder=0,
                        label='Cumulative error', ds='steps-post')
                ax.fill_between(self._times, np.ones(r.shape) * y[0], r,
                                color='tab:red', alpha=0.03, zorder=0,
                                step='post')
                ax.set_ylim(y)

        ax1.legend(loc='upper right')
        return fig


# Create model
print('Creating model')
m = Model(model_file)
if model_x0:
    x0 = m.parameter_values()

# Create problems
print('Loading data')
problems = []
for experiment, serieses in data.items():
    for series in serieses:
        problems.append(DataLogProblem(m, experiment, series))
        #problems[-1].figure(m.parameter_values())
        if debug:
            break
    if debug:
        break

# Create error
print(f'Creating error from {len(problems)} problems')
errors = [pints.MeanSquaredError(p) for p in problems]
error = pints.SumOfErrors(errors)


class IgnoringError(pints.ErrorMeasure):
    def __init__(self, error):
        self._e = error
    def __call__(self, parameters):
        try:
            return self._e(parameters)
        except myokit.SimulationError:
            return np.inf
    def n_parameters(self):
        return self._e.n_parameters()


error = IgnoringError(error)


# Load stored result or optimise
if not redraw:
    # Optimise
    print('Creating optimisation controller')
    opt = pints.OptimisationController(
        error, x0, boundaries=boundaries, method=pints.CMAES)
    opt.set_max_iterations(1 if debug else None)
    opt.set_parallel(not debug)
    opt.set_max_unchanged_iterations(100)

    # Run optimisation
    with np.errstate(all='ignore'):
        x1, e1 = opt.run()

    # Store parameters
    if not os.path.isdir(root):
        os.mkdir(root)
    path = f'{root}/final.txt'
    print(f'Storing parameters in {path}')
    with open(path, 'w') as f:
        for v in x1:
            print(f'  {v}')
            f.write(f'{myokit.float.str(v, full=True)}\n')
    path = f'{root}/error.txt'
    print(f'Storing error in {path}')
    with open(path, 'w') as f:
        f.write(f'{myokit.float.str(e1, full=True)}\n')

else:
    # Load stored results
    print('Loading stored result')
    path = f'{root}/final.txt'
    with open(path, 'r') as f:
        x1 = [float(line) for line in f.readlines()]
    path = f'{root}/error.txt'
    with open(path, 'r') as f:
        e1 = [float(f.read())][0]


# Store goodness of fit figures
print('Creating figures')
for p, e in zip(problems, errors):
    p.figure(x1, x0, total_error_value=e1, individual_error=e)
