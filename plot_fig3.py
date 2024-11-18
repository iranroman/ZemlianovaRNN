
import numpy as np
import matplotlib.pyplot as plt
from zrnn import analysis


def _get_plots_abc(axs, iccs=(-1, 0.5, 1.8)):
    y0 = [0.075, 0.075, -0.01]
    params = analysis.DEF_PARAMS
    time = np.linspace(0, 0.5, int(0.5 * 1e3 + 1))
    for i, (icc, ax) in enumerate(zip(iccs, axs)):
        params["I_cc"] = icc
        solution = analysis.solver(params, y0, 0.5)
        ax.plot(time, solution)
        ax.set_xlabel('Time (ms)', fontsize=16)
        ax.set_ylabel('Activity', fontsize=16)
        ax.set_title(f'I_cc = {icc}', fontsize=16)
        ax.set_ylim([-0.2, 0.2])
        ax.set_xlim([time[0], time[-1]])


def _get_plot_d(ax, icc_i = 0.19, icc_f = 1):
    i_cc_range = np.linspace(icc_i, icc_f, 100)
    y0 = [0.075, 0.075, -0.01]
    params = analysis.DEF_PARAMS
    periods = []
    for icc in i_cc_range:
        params["I_cc"] = icc
        solution = analysis.solver(params, y0, 5)
        periods.append(analysis.get_oscillation_period(solution))
    ax.plot(i_cc_range, periods, c='g')
    ax.set_xlim([0, 1]), ax.set_ylim([0, 1000])
    ax.set_xlabel('Context Cue (I_cc)', fontsize=16)
    ax.set_ylabel('Oscillation Period [ms]', fontsize=16)


def _get_plot_e(ax):
    i_cc_range = np.linspace(-2, 2, 1000)
    y0 = [0.075, 0.075, -0.01]
    params = analysis.DEF_PARAMS
    fpss = []
    osc = []
    for icc in i_cc_range:
        params["I_cc"] = icc

        model = analysis.DynamicalSystem(params)
        fps, stability, spirality = model.get_fixed_points(100)
        fpss.append(fps)
        cmap = {0: 'k', 1: 'r'}
        for fp, t, p in zip(fps, stability, spirality):
            ax.plot(icc, fp[-1], 'o', c=cmap[t])
            if not t and len(fps) < 2:
                osc.append(icc)
                solution = analysis.solver(params, y0, 2)
                z_max, z_min = solution[1000:, -1].max(), solution[1000:, -1].min()
                ax.plot([icc] * 2, [z_max, z_min], 'o', c='g')
    i_hc = min(osc)
    i_hb = max(osc)
    ax.axvline(x=i_hc, ls='--', c='k')
    ax.axvline(x=i_hb, ls='--', c='k')
    ax.axvline(x=-1, ls=':', c='k')
    ax.axvline(x=0.5, ls=':', c='k')
    ax.axvline(x=1.8, ls=':', c='k')
    ax.set_xlabel('Context Cue (I_cc)', fontsize=16)
    ax.set_ylabel('Int-I (z)', fontsize=16)


def _get_plot_f(ax, iccs=(0.197, 0.3, 0.5, 0.8, 1.2)):
    import matplotlib as mpl
    y0 = [0.075, 0.075, -0.01]
    params = analysis.DEF_PARAMS
    cmap = mpl.colormaps["PuBuGn"]
    c = 1.
    for icc in iccs:
        params["I_cc"] = icc
        solution = analysis.solver(params, y0, 5)
        period = analysis.get_oscillation_period(solution)
        to_plot = solution[500:500 + int(period)]
        ax.plot(to_plot[:, 0], to_plot[:, -1], 'o', c=cmap(c))
        c -= .2
    ax.set_xlabel('Tap-E (x)', fontsize=16)
    ax.set_ylabel('Int-I (z)', fontsize=16)
    ax.set_xlim([-.2, .2]), ax.set_ylim([-.2, .2])


def main():
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(1000/96, 1000/96), dpi=96)
    _get_plots_abc(axs[0])
    _get_plot_d(axs[1, 0])
    _get_plot_e(axs[1, 1])
    _get_plot_f(axs[1, 2])
    plt.tight_layout()
    plt.savefig('plots/fig_3.png', dpi=96)


if __name__ == '__main__':
    main()
