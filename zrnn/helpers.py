
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from zrnn.models import ZemlianovaRNN
from zrnn.datasets import generate_stimuli


def relu(x: np.typing.ArrayLike) -> np.typing.ArrayLike:
    return np.maximum(x, 0)


def normalize(x: np.typing.ArrayLike) -> np.typing.ArrayLike:
    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)
    range_val = max_val - min_val
    return np.divide(x - min_val, range_val, out=np.zeros_like(x), where=range_val > 0)


def standardize(x: np.typing.ArrayLike) -> np.typing.ArrayLike:
    from scipy.stats import zscore
    return zscore(x, axis=0)


def load_model(model_path: str | Path, config_path: Path | str = 'config.yaml', **kwargs) -> ZemlianovaRNN:
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZemlianovaRNN(**config['model']).to(device)
    map_location = kwargs.get('map_location', device)  # because I trained on mps
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=map_location))
    model.eval()
    return model


def generate_stimuli_from_cc(context_cue_amplitude: float, t_onset_ms: float = .05, *args, **kwargs):
    period = .1 / context_cue_amplitude
    return torch.tensor(np.array([generate_stimuli(period, t_onset_ms, *args, **kwargs)[1:3]]),
                        dtype=torch.float).swapaxes(1, 2)


def drive_model(model: ZemlianovaRNN,
                context_cue_amplitude: float,
                time_steps_ms: int = 52_000,
                discard_steps: int = 2_000,
                initial_neuron_activity: torch.Tensor = None,
                t_onset_ms: float = .05,
                device: str = 'cpu',
                **kwargs):
    model.to(device)
    rnn_input_tensor = generate_stimuli_from_cc(context_cue_amplitude,
                                                t_onset_ms=t_onset_ms,
                                                duration=52,
                                                continuation_phase=kwargs.get('continuation_phase', True)).to(device)
    if initial_neuron_activity is None:
        neuron_state = model.initHidden(1).to(device)
    else:
        neuron_state = initial_neuron_activity.to(device)
    model_outputs, neuron_states = [], []
    with torch.inference_mode():
        for t in range(time_steps_ms):
            output, neuron_state = model(rnn_input_tensor[:, t, :], neuron_state)
            model_outputs.append(output.detach().cpu().numpy())
            if t >= discard_steps:
                neuron_states.append(neuron_state.cpu().numpy())
    neuron_states = np.array(neuron_states).squeeze(1)
    model_outputs = np.array(model_outputs).squeeze()
    valid_outputs = model_outputs[discard_steps:]
    return valid_outputs, neuron_states


def get_peaks(signal: np.ndarray | torch.Tensor) -> np.ndarray:
    # Since no noise in this part, no need for filtering
    from scipy.signal import find_peaks
    return find_peaks(signal, height=0)[0]


def get_neuron_phases_data(model_output: np.typing.ArrayLike, neurons_activity: np.typing.ArrayLike) -> pd.DataFrame:
    # Return the relative phase of each neuron wrt to the tap in the first ITI, along with other information

    # Normalize and retrieve inter tap interval indices
    normalized_neurons_activity = normalize(relu(neurons_activity))
    tap_indices = get_peaks(model_output)
    inter_tap_interval = slice(*tap_indices[1:3])
    inter_tap_interval_activity = normalized_neurons_activity[inter_tap_interval]
    # Calculate the phase by finding the shortest period between neuron max and tap time
    max_args = np.argmax(inter_tap_interval_activity, axis=0)
    neuron_phases = np.minimum(max_args, inter_tap_interval_activity.shape[0] - max_args)
    neuron_phases = neuron_phases / (inter_tap_interval.stop - inter_tap_interval.start)
    df = pd.DataFrame({'phase': neuron_phases,
                       'n_type': [1 if i < .8 * len(neuron_phases)
                                  else 0
                                  for i in range(len(neuron_phases))]})
    # Determine if tap or inter-tap neuron
    df['tap'] = ((df.phase > 0.) & (df.phase < .2)).astype(int)

    def _get_sub_pop(row: pd.Series) -> int:
        # Divides the neuron to subgroups. The outcome depends on the context cue - is that how it should be?
        if bool(row.tap) and bool(row.n_type):
            return 0
        if bool(row.tap) and not bool(row.n_type):
            return 1
        if not bool(row.tap) and not bool(row.n_type):
            return 2
        if not bool(row.tap) and bool(row.n_type):
            return 3

    # Determine the subgroup
    df['sub_pop'] = [_get_sub_pop(row) for _, row in df.iterrows()]
    return df


SUB_POPS = {0: 'tap_E', 1: 'tap_I', 2: 'inter_tap_I', 3: 'inter_tap_E'}


def generate_initial_neuron_activity(neuron_phases_data: pd.DataFrame, random_seed: int = 112023) -> torch.Tensor:
    generator = torch.Generator().manual_seed(random_seed)
    # for the general neuron population, draw numbers from uniform(-1, 1)
    initial_neuron_activity = 2 * torch.rand(len(neuron_phases_data), generator=generator) - 1
    # Single out the tap neuron population and draw from uniform (0, 1) to all of them:
    tap_neurons = neuron_phases_data['sub_pop'].isin([0, 1])
    initial_neuron_activity[tap_neurons] = torch.rand(1, generator=generator)
    return initial_neuron_activity.unsqueeze(0)


def get_principal_components(model: ZemlianovaRNN,
                             context_cue_range: (float, float) = (-1, .2),
                             num_trajectories: int = 100,
                             phases_context_cue_amplitude: float = .5,
                             random_seed: int = 112023
                             ) -> (PCA, pd.DataFrame):
    # Calculate the activities for context cue in the oscillatory regime to get the subgroups
    model_output, neurons_activity = drive_model(model, phases_context_cue_amplitude)
    neuron_phases = get_neuron_phases_data(model_output, neurons_activity)
    # Now calculate the activities in the non-oscillatory regime
    all_activities = np.concat([drive_model(model, cc,
                                            time_steps_ms=500,
                                            discard_steps=0,
                                            initial_neuron_activity=generate_initial_neuron_activity(neuron_phases,
                                                                                                     random_seed))[1]
                                for cc in np.linspace(*context_cue_range, num_trajectories)])
    # Normalize and fit to the PCA. Not certain about the way of normalization here
    all_activities = standardize(all_activities)
    pca = PCA(n_components=len(neuron_phases))
    pca.fit(all_activities)
    return pca, neuron_phases


def get_vector_field(model: ZemlianovaRNN,
                     context_cue_amplitude: float,
                     initial_neuron_activity: torch.Tensor,
                     trained_pca: PCA,
                     span: (float, float) = (-5, 5),
                     center: (float, float) = (0, 0),
                     time_steps_ms: int = 500,
                     grid_length: int = 256,
                     at_time: int = None,
                     device: str = 'cpu') -> dict:
    from scipy.linalg import norm
    model.to(device)
    c0_points = torch.linspace(*span, grid_length).to(device) + center[0]
    c1_points = torch.linspace(*span, grid_length).to(device) + center[1]
    grid_c0, grid_c1 = torch.meshgrid(c0_points, c1_points, indexing='xy')
    stimulus = generate_stimuli_from_cc(context_cue_amplitude,
                                        continuation_phase=True).to(device)
    i_proj = torch.tensor(trained_pca.transform(initial_neuron_activity), dtype=torch.float32).to(device)
    i_proj = torch.broadcast_to(i_proj, [grid_length, grid_length, 1, i_proj.shape[-1]]).clone()
    dc0, dc1, speeds = [], [], []
    for t in range(time_steps_ms):
        # broadcast the initial conditions to the grid on the projected plane
        i_proj[:, :, 0, :2] = torch.stack([grid_c0, grid_c1], dim=2)
        # transform back to the original RNN space and forward one step
        i_orig = _transform_tensor(i_proj, trained_pca, inverse=True).to(device)
        with torch.inference_mode():
            _, i_orig = model(stimulus[:, t, :], i_orig)
        # Project back to the PC plane to get i_proj at t + 1, calculate the gradient, record and reassign
        i_proj_tp1 = _transform_tensor(i_orig, trained_pca)
        grad = (i_proj_tp1 - i_proj.cpu())
        dc0.append(grad[:, :, 0, 0].numpy()), dc1.append(grad[:, :, 0, 1].numpy())
        speeds.append(np.squeeze(norm(grad, axis=3), axis=2))
        i_proj = i_proj_tp1.to(device)
    out_slice = slice(None) if at_time is None else at_time
    return {"dc0": np.stack(dc0)[out_slice],
            "dc1": np.stack(dc1)[out_slice],
            "speed": np.stack(speeds)[out_slice]}



def _transform_tensor(tensor:torch.Tensor, pca: PCA, inverse: bool = False) -> torch.Tensor:
    # Performs PCA transformation on arbitrary tensor of vectors in the RNN / projected plane
    # (as sklearn is incompatible with tensors)
    size = tensor.shape
    shrank = [size[0] * size[0], size[-1]]
    out_func = lambda f: torch.tensor(f(tensor.reshape(shrank).cpu().detach().numpy()), dtype=torch.float32).reshape(size)
    return out_func(pca.transform) if not inverse else out_func(pca.inverse_transform)


# def get_vector_field(model: ZemlianovaRNN,
#                      context_cue_amplitude: float,
#                      initial_neuron_activity: torch.Tensor,
#                      trained_pca: PCA,
#                      bounds: (float, float),
#                      time_steps_ms: int = 500,
#                      grid_length: int = 16,
#                      at_time: int = None) -> dict:
#     c0_points = np.linspace(bounds[0], bounds[1], grid_length)
#     c1_points = np.linspace(bounds[0], bounds[1], grid_length)
#     grid_c0, grid_c1 = np.meshgrid(c0_points, c1_points)
#     stimulus = generate_stimuli_from_cc(context_cue_amplitude, continuation_phase=True).to(next(model.parameters()).device)
#     output_shape = [grid_length, grid_length, time_steps_ms]
#     dc0, dc1, speed = np.empty(output_shape), np.empty(output_shape), np.empty(output_shape)
#     it = np.nditer(np.empty_like(grid_c0), flags=['multi_index'])
#     while not it.finished:
#         _dc0, _dc1, _speed = get_vector_dynamics_on_point(model,
#                                                           [grid_c0[it.multi_index], grid_c1[it.multi_index]],
#                                                           stimulus,
#                                                           initial_neuron_activity,
#                                                           trained_pca,
#                                                           time_steps_ms,)
#         dc0[it.multi_index] = _dc0
#         dc1[it.multi_index] = _dc1
#         speed[it.multi_index] = _speed
#         it.iternext()
#         out_slice = slice(None) if at_time is None else at_time
#     return {"dc0": np.moveaxis(dc0, 2, 0)[out_slice],
#             "dc1": np.moveaxis(dc1, 2, 0)[out_slice],
#             "speed": np.moveaxis(speed, 2, 0)[out_slice]}
#
#
# def get_vector_dynamics_on_point(model: ZemlianovaRNN,
#                                  point: (float, float),
#                                  stimulus: torch.Tensor,
#                                  initial_neuron_activity: torch.Tensor,
#                                  trained_pca: PCA,
#                                  time_steps_ms: int = 500,) -> (np.ndarray, np.ndarray, np.ndarray):
#     from scipy.linalg import norm
#     i_proj_t = trained_pca.transform(initial_neuron_activity)
#     d0, d1, speeds = [], [], []
#     for t in range(time_steps_ms):
#         with torch.inference_mode():
#             i_proj_t[0, :2] = point
#             i_orig_t = torch.tensor((trained_pca.inverse_transform(i_proj_t)), dtype=torch.float32)
#             _, i_orig_t = model(stimulus[:, t, :], i_orig_t)
#             tmp = trained_pca.transform(i_orig_t)
#             grad_t = (tmp - i_proj_t)[0]
#             d0.append(grad_t[0]), d1.append(grad_t[1])
#             speeds.append(norm(grad_t))
#             i_proj_t = tmp
#     # return_slice = slice(None) if at_time is None else slice(at_time, at_time + 1)
#     # if at_time < 0:
#     #     return_slice = slice(time_steps_ms - 1, time_steps_ms)
#     return (np.array(d0, dtype=np.float32),
#             np.array(d1, dtype=np.float32),
#             np.array(speeds, dtype=np.float32))
