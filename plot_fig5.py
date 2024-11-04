
from pathlib import Path
import numpy as np
from zrnn import utils
import matplotlib.pyplot as plt
from argparse import ArgumentParser


CONTEXT_CUES = (-.5, -.2, .1, 1., 1.5, 2.)
TIMES_CONTEXT_CUE = .5
TIMES = (80, 200, 400)
TIMES = (40, 90, 180)


def _load_data(cc_enum: int, load_dir: Path):
    files = load_dir.glob(f'*_{cc_enum}.npy')
    data = {}
    for file in files:
        name = file.stem.split('_')[0]
        data[name] = np.load(file)
    return data


def _gen_single_plot(data: dict,
                     ax: plt.Axes, span: (float, float),
                     grid_res: int = 256,
                     center_around_end_point: bool = False):
    end_point = 0., 0.
    if center_around_end_point:
        end_point = (data['trajectory0'][-1], data['trajectory1'][-1])
    c0_points = np.linspace(*span, grid_res) + end_point[0]
    c1_points = np.linspace(*span, grid_res) + end_point[1]
    grid_c0, grid_c1 = np.meshgrid(c0_points, c1_points)
    ax.streamplot(grid_c0, grid_c1, data['dc0'], data['dc1'], color='k')
    ax.plot(data['trajectory0'], data['trajectory1'], alpha=.5, c='r', linewidth=3)
    x_lims = (span[0] + end_point[0]), (span[1] + end_point[0])
    y_lims = (span[0] + end_point[1]), (span[1] + end_point[1])
    ax.set_xlim(*x_lims), ax.set_ylim(*y_lims)
    return ax.pcolormesh(grid_c0, grid_c1, data['speed'], cmap='Spectral_r')


def _add_cbar(cplot, fig):
    fig.subplots_adjust(right=.8)
    cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
    fig.colorbar(cplot, cax=cbar_ax)
    cbar_ax.yaxis.set_label_position('right')
    cbar_ax.set_ylabel('Speed', rotation=270, labelpad=20, fontsize=16)
    cbar_ax.yaxis.set_ticks_position('left')


def _cc_plot(context_cues: [float], span: (float, float), grid_res: int):
    fig, axs = plt.subplots(nrows=len(CONTEXT_CUES) // 3, ncols=3, figsize=(1000/96, 1000/96), dpi=96)
    load_dir = Path('fig_5_data/plot_a')
    for i, cc in enumerate(context_cues):
        idxs = (i // 3, i % 3)
        ax = axs[i // 3, i % 3]
        data = _load_data(i, load_dir)
        im = _gen_single_plot(data, ax, span, grid_res, center_around_end_point=True)
        ax.set_title(f'CC = {cc}')
        if idxs == ((len(context_cues) - 1) // 3, 0):
            ax.set_xlabel('PC1', fontsize=16), ax.set_ylabel('PC2', fontsize=16)
        if i == len(context_cues) - 1:
            _add_cbar(im, fig)
    fig.savefig('plots/fig_5a.png', dpi=96)


def _times_plot(times: [int], span: (float, float), grid_res: int):
    fig, axs = plt.subplots(ncols=len(TIMES), figsize=(1000/96, 500/96), dpi=96)
    load_dir = Path('fig_5_data/plot_b')
    for i, time in enumerate(times):
        ax = axs[i]
        data = _load_data(i, load_dir)
        im = _gen_single_plot(data, ax, span, grid_res)
        ax.set_title(f'time = {time} [ms]')
        if i == 0:
            ax.set_xlabel('PC1', fontsize=16), ax.set_ylabel('PC2', fontsize=16)
        if i == len(times) - 1:
            _add_cbar(im, fig)
    # TODO add inner subgroups plot
    fig.subplots_adjust(bottom=0.175)
    fig.savefig('plots/fig_5b.png', dpi=96)


def _gen_cc_plot_data(model,
                      pca,
                      initial_neurons_activity,
                      context_cues: [float],
                      span: (float, float) = (-5, 5),
                      grid_res: int = 256,
                      device: str = 'cpu'):
    save_dir = Path('fig_5_data/plot_a')
    save_dir.mkdir(exist_ok=True, parents=True)
    for i, cc in enumerate(context_cues):
        _, neuron_activities = helpers.drive_model(model,
        _, neuron_activities = utils.drive_model(model,
                                                         cc,
                                                         time_steps_ms=500,
                                                         discard_steps=0,
                                                         initial_neuron_activity=initial_neurons_activity,
                                                         device=device)
        trajectory_0, trajectory_1 = pca.transform(neuron_activities).T[:2]
        end_point = trajectory_0[-1], trajectory_1[-1]
        vec_field = utils.get_vector_field(model, cc, initial_neurons_activity, pca, span, center=end_point,
                                                   grid_res=grid_res, at_time=-1, device=device)
        vec_field.update({"trajectory0": trajectory_0, "trajectory1": trajectory_1})
        for k, v in vec_field.items():
            np.save(save_dir.joinpath(f'{k}_{i}.npy'), v)


def _gen_times_plot_data(model,
                         pca,
                         times: [int],
                         context_cue_amplitude: float,
                         span: (float, float) = (-5, 5),
                         grid_res: int = 256,
                         device: str = 'cpu'):
    save_dir = Path('fig_5_data/plot_b')
    save_dir.mkdir(exist_ok=True, parents=True)
    initial_neurons_activity = model.initHidden(1)
    _, neuron_activities = utils.drive_model(model,
                                                     context_cue_amplitude,
                                                     time_steps_ms=times[-1],
                                                     discard_steps=0,
                                                     initial_neuron_activity=initial_neurons_activity,
                                                     device=device)
    trajectory_0, trajectory_1 = pca.transform(neuron_activities).T[:2]
    end_point = 0, 0
    vec_fields = utils.get_vector_field(model, context_cue_amplitude, initial_neurons_activity, pca, span,
                                                center=end_point,
                                                time_steps_ms=times[-1], grid_res=grid_res, device=device)

    for i, time in enumerate(times):
        tmp_field = {k: v[time - 1] for k, v in vec_fields.items()}
        tmp_trajectory_0, tmp_trajectory_1 = trajectory_0[:time], trajectory_1[:time]
        tmp_field.update({"trajectory0": tmp_trajectory_0, "trajectory1": tmp_trajectory_1})
        for k, v in tmp_field.items():
            np.save(save_dir.joinpath(f'{k}_{i}.npy'), v)


def _args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--type', type=str, default='both',
                        choices=['both', 'times', 'cc'],
                        help='Types of plots to generate, both / times / cc. Defaults to both',
                        nargs=1,
                        metavar='PLOT_TYPE')
    parser.add_argument('-s', '--span', type=float, default=(-5, 5), nargs=2,
                        help='Span of the axes of the plot in the RNN space',
                        metavar=('LOWER_BOUND', 'UPPER_BOUND'))
    parser.add_argument('-g', '--grid_res', type=int, default=128,
                        help='Grid resolution of the plot. RES ** 2 points will be generated',
                        metavar='RES')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device to run the model on')
    parser.add_argument('-cc', '--context_cues', type=float, default=CONTEXT_CUES, nargs='+',
                        help='Context to generate plots on (figure 5a in the paper)')
    parser.add_argument('-t', '--times', type=int, default=TIMES, nargs='+',
                        help='Times to generate plots on (figure 5b in the paper)',
                        metavar='TIME_1')
    parser.add_argument('-tcc', '--times_context_cue', type=float, default=.5, nargs=1,
                        help='The context cue the times plot will be generated on')
    parser.add_argument('--config', type=str, default='fig5_config.yaml',
                        help='path to config file')
    parser.add_argument('--model', type=str, default='best_model.pth',
                        help='path to model pth file')
    return parser


def main():
    args = _args().parse_args()
    model = helpers.load_model(model_path=args.model, config_path=args.config)
    pca, phases_df = helpers.get_principal_components(model)
    initial_conditions = helpers.generate_initial_neuron_activity(phases_df)
    model = utils.load_model(model_path=args.model, config_path=args.config)
    pca, phases_df = utils.get_principal_components(model)
    initial_conditions = utils.generate_initial_neuron_activity(phases_df)
    const_args = dict(
        span=args.span,
        grid_res=args.grid_res,
        device=args.device,
    )
    if args.type in ('both', 'cc'):
        _gen_cc_plot_data(model,
                          pca,
                          initial_conditions,
                          context_cues=args.context_cues,
                          **const_args)
        _cc_plot(context_cues=args.context_cues,
                 span=args.span,
                 grid_res=args.grid_res,)
    if args.type in ('times', 'both'):
        _gen_times_plot_data(model, pca, times=args.times, context_cue_amplitude=args.times_context_cue, **const_args)
        _times_plot(times=args.times,
                    span=args.span,
                    grid_res=args.grid_res)



if __name__ == '__main__':
    main()
