
from pathlib import Path
import numpy as np
from zrnn import helpers
import matplotlib.pyplot as plt
from fire import Fire


CONTEXT_CUES = (-.5, -.2, .1, 1., 1.5, 2.)
TIMES_CONTEXT_CUE = .5
TIMES = (80, 200, 400)


def _load_data(cc_enum: int, load_dir: Path):
    files = load_dir.glob(f'*_{cc_enum}.npy')
    data = {}
    for file in files:
        name = file.stem.split('_')[0]
        data[name] = np.load(file)
    return data


def _gen_single_plot(data: dict, ax: plt.Axes, span: (float, float), grid_length: int = 256, center_around_end_point: bool = False):
    end_point = 0., 0.
    if center_around_end_point:
        end_point = (data['trajectory0'][-1], data['trajectory1'][-1])
    c0_points = np.linspace(*span, grid_length) + end_point[0]
    c1_points = np.linspace(*span, grid_length) + end_point[1]
    grid_c0, grid_c1 = np.meshgrid(c0_points, c1_points)
    ax.streamplot(grid_c0, grid_c1, data['dc0'], data['dc1'], color='k')
    ax.plot(data['trajectory0'], data['trajectory1'], alpha=.5, c='r', linewidth=3)
    x_lims = (span[0] + end_point[0]), (span[1] + end_point[0])
    y_lims = (span[0] + end_point[1]), (span[1] + end_point[1])
    ax.set_xlim(*x_lims), ax.set_ylim(*y_lims)
    return ax.pcolormesh(grid_c0, grid_c1, data['speed'], cmap='Spectral_r')


def _plot_a(span: (float, float), grid_length: int, context_cues: [float] = CONTEXT_CUES):
    fig, axs = plt.subplots(nrows=len(CONTEXT_CUES) // 3, ncols=3, figsize=(1000/96, 1000/96), dpi=96)
    load_dir = Path('fig_5_data/plot_a')
    for i, cc in enumerate(context_cues):
        ax = axs[i // 3, i % 3]
        data = _load_data(i, load_dir)
        im = _gen_single_plot(data, ax, span, grid_length, center_around_end_point=True)
        ax.set_title(cc)
        if i == len(CONTEXT_CUES) - 1:
            plt.colorbar(im)
    fig.tight_layout()
    fig.savefig('plots/fig_5a.png', dpi=96)


def _plot_b(span: (float, float), grid_length: int, times: [int] = TIMES):
    fig, axs = plt.subplots(ncols=len(TIMES), figsize=(1000/96, 333/96), dpi=96)
    load_dir = Path('fig_5_data/plot_b')
    for i, time in enumerate(times):
        ax = axs[i]
        data = _load_data(i, load_dir)
        im = _gen_single_plot(data, ax, span, grid_length)
        ax.set_title(time)
        if i == len(TIMES):
            plt.colorbar(im)
    # TODO add inner subgroups plot
    fig.tight_layout()
    fig.savefig('plots/fig_5b.png', dpi=96)


def _gen_plot_a_data(model,
                     pca,
                     initial_neurons_activity,
                     context_cues: [float] = CONTEXT_CUES,
                     span: (float, float) = (-5, 5),
                     grid_length: int = 256,
                     device: str = 'cpu'):
    save_dir = Path('fig_5_data/plot_a')
    save_dir.mkdir(exist_ok=True, parents=True)
    for i, cc in enumerate(context_cues):
        _, neuron_activities = helpers.drive_model(model,
                                                         cc,
                                                         time_steps_ms=500,
                                                         discard_steps=0,
                                                         initial_neuron_activity=initial_neurons_activity,
                                                         device=device)
        trajectory_0, trajectory_1 = pca.transform(neuron_activities).T[:2]
        end_point = trajectory_0[-1], trajectory_1[-1]
        vec_field = helpers.get_vector_field(model, cc, initial_neurons_activity, pca, span, center=end_point,
                                                   grid_length=grid_length, at_time=-1, device=device)
        vec_field.update({"trajectory0": trajectory_0, "trajectory1": trajectory_1})
        for k, v in vec_field.items():
            np.save(save_dir.joinpath(f'{k}_{i}.npy'), v)


def _gen_plot_b_data(model,
                     pca,
                     initial_neurons_activity,
                     times: [int] = TIMES,
                     context_cue_amplitude: float = TIMES_CONTEXT_CUE,
                     span: (float, float) = (-5, 5),
                     grid_length: int = 256,
                     device: str = 'cpu'):
    save_dir = Path('fig_5_data/plot_b')
    save_dir.mkdir(exist_ok=True, parents=True)
    _, neuron_activities = helpers.drive_model(model,
                                                     context_cue_amplitude,
                                                     time_steps_ms=times[-1],
                                                     discard_steps=0,
                                                     initial_neuron_activity=initial_neurons_activity,
                                                     device=device)
    trajectory_0, trajectory_1 = pca.transform(neuron_activities).T[:2]
    # end_point = trajectory_0[-1], trajectory_1[-1]
    end_point = 0 ,0
    vec_fields = helpers.get_vector_field(model, context_cue_amplitude, initial_neurons_activity, pca, span,
                                                center=end_point,
                                                time_steps_ms=times[-1], grid_length=grid_length, device=device)

    for i, time in enumerate(times):
        tmp_field = {k: v[time - 1] for k, v in vec_fields.items()}
        tmp_trajectory_0, tmp_trajectory_1 = trajectory_0[:time], trajectory_1[:time]
        tmp_field.update({"trajectory0": tmp_trajectory_0, "trajectory1": tmp_trajectory_1})
        for k, v in tmp_field.items():
            np.save(save_dir.joinpath(f'{k}_{i}.npy'), v)


def main(gen_data: bool = False, span: (float, float) = (-5, 5), grid_length: int = 256, device: str = 'cpu'):
    if gen_data:
        model = helpers.load_model('best_model.pth', 'fig5_config.yaml')
        pca, phases_df = helpers.get_principal_components(model)
        initial_conditions = helpers.generate_initial_neuron_activity(phases_df)
        _gen_plot_a_data(model, pca, initial_conditions, span=span, grid_length=grid_length, device=device)
        _gen_plot_b_data(model, pca, initial_conditions, span=span, grid_length=grid_length, device=device)
    _plot_a(span=span, grid_length=grid_length)
    _plot_b(span=span, grid_length=grid_length)


if __name__ == '__main__':
    Fire(main)
