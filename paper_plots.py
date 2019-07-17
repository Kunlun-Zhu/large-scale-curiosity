import os
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Experiments(object):
    def __init__(self, directory, reload_logs=False):
        self.directory = directory
        _, self.description = os.path.split(self.directory)
        self.all_experiment_names = os.listdir(self.directory)
        self.all_experiment_names = [e for e in self.all_experiment_names if
                                     os.path.isdir(os.path.join(self.directory, e))]
        self.experiments = [Experiment(os.path.join(self.directory, e),
                                       reload_log=reload_logs) for e in self.all_experiment_names]
        self.grouped_experiments = self.group_experiments()

    def group_experiments(self):
        groups = {}
        for e in self.experiments:
            if e.env_name not in groups:
                groups[e.env_name] = {}
            if e.method not in groups[e.env_name]:
                groups[e.env_name][e.method] = MultipleSeedsExperiment([])
            groups[e.env_name][e.method].add(e)
        return groups


class MultipleSeedsExperiment(object):
    def __init__(self, experiments):
        self.experiments = experiments

    def add(self, experiment):
        self.experiments.append(experiment)

    def get_xs_ys(self, y_series='eprew_recent'):
        list_of_xs, list_of_ys, n_steps_per_update = [], [], []
        for exp in self.experiments:
            list_of_xs.append(exp.timeseries('n_updates'))
            list_of_ys.append(exp.timeseries(y_series))

            n_steps_per_update.append(exp.timeseries('tcount')[-1] / exp.timeseries('n_updates')[-1])
        x, y = group_timeseries(list_of_xs, list_of_ys)
        return x * np.max(n_steps_per_update), y


class Experiment(object):
    log_separator = '------------'

    def __init__(self, directory, reload_log):
        self.directory = directory
        _, self.exp_name = os.path.split(self.directory)
        self.env_name, self.method, self.seed = self.parse_name(name=self.exp_name)
        self.log = self.load_log(regenerate=reload_log)
        # self.exp_details = self.load_exp_info()
        self._timeseries = {}

    def parse_name(self, name):
        if name.startswith('mariogen'):
            return self.parse_name_mariogen(name)
        elif name == 'mz8gpu':
            return 'MontezumaRevenge', 'idf', 0
        elif name == 'mario16gpurf-00':
            return 'Mario', 'rf_large', 0
        elif name == 'mario1gpurf':
            return 'Mario', 'rf_small', 0
        else:
            if '_' in self.exp_name:
                return self.parse_name_yura_convention(name)
            else:
                return self.parse_name_harri_convention(name)


    def parse_name_yura_convention(self, name):
        env, method, seed, *extra = name.split('_')
        if method == 'randfeat':
            method = 'rf'
        if extra != []:
            method += '_' + '_'.join(extra)
        return env, method, seed

    def parse_name_harri_convention(self, name):
        envs = {'mz': 'MontezumaRevenge',
                'mario': 'Mario',
                'breakout': 'Breakout',
                'pong': 'Pong',
                'spaceinvaders': 'SpaceInvaders',
                'qbert': 'Qbert',
                'beamrider': 'BeamRider',
                'riverraid': 'Riverraid',
                'seaquest': 'Seaquest',
                'unitysparse': 'Unity-Sparse',
                'unitytv0': 'Unity-TV0',
                'unitytv2': 'Unity-TV2',
                'multipong': "Multiplayer-Pong",
                'juggling': "Roboschool-Juggling"
                }
        methods = ['vaesph', 'pix2pixbn', 'pix2pix', 'idfbnnontrainable', 'idfbntrainable', 'idf', 'rf', 'vaenonsph',
                   'randfeatbn', 'ext']  # the order matters
        method = [m for m in methods if m in name]
        method = method[0]
        seed = int(name[-1]) - 1
        env = name[:-(1 + len(method))]
        env = envs[env]
        return env, method, seed

    def parse_name_mariogen(self, name):
        name = name[len('mariogen'):]
        env = 'MarioGen'
        if name.startswith('fixed'):
            extra_method = '_fixed'
            name = name[len('fixed'):]
        else:
            extra_method = ''
        method = name[:len('1to1')]
        name = name[len('1to1'):]
        assert name.startswith('idf') or name.startswith('rf'), self.directory
        if name.startswith('idf'):
            feat_method = 'idf'
        elif name.startswith('rf'):
            feat_method = 'rf'
        method += '_{}'.format(feat_method)
        method += extra_method
        name = name[len(feat_method):]
        seed = int(name) - 1
        return env, method, seed

    def load_exp_info(self):
        directory = os.path.join(self.directory, 'exp_details.pkl')
        if os.path.exists(directory):
            with open(directory, 'rb') as f:
                try:
                    exp_details = pickle.load(f)
                    if not exp_details:
                        exp_details = None
                except:
                    exp_details = None
        else:
            exp_details = None
        return exp_details

    def load_log(self, regenerate=False):
        pickle_of_log = os.path.join(self.directory, 'log.pickle')
        if os.path.exists(pickle_of_log) and not regenerate:
            with open(pickle_of_log, 'rb') as f:
                print('loading log ', pickle_of_log)
                log = pickle.load(f)
        else:
            print('parsing log {}'.format(self.directory))
            log_file = os.path.join(self.directory, 'log.txt')

            log = self.parse_log_from_file(log_file)
            with open(pickle_of_log, 'wb') as f:
                pickle.dump(log, f, protocol=-1)
        return log

    def parse_log_from_file(self, log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines if l.startswith('|') or l.startswith(Experiment.log_separator)]
        chunks = []
        current_chunk = {}
        for l in lines:
            if l.startswith(Experiment.log_separator):
                if current_chunk != {}:  # end of chunk
                    chunks.append(current_chunk)
                current_chunk = {}
            else:
                name, value = [p.strip() for p in l.split('|') if p.strip() != '']
                try:
                    if value != 'nan':
                        value = eval(value)
                    else:
                        value = np.nan
                except:
                    print(name, value, "failed")
                current_chunk[name] = value
        return chunks

    def timeseries(self, name):
        assert self.log is not None
        if self._timeseries.get(name, None) is None:
            self._timeseries[name] = [entry.get(name, 0.) for entry in self.log]
        return self._timeseries[name]

    def __str__(self):
        return self.exp_name


def group_timeseries(list_of_xs, list_of_ys):
    grouped_values = {}
    for xs, ys in zip(list_of_xs, list_of_ys):
        for x, y in zip(xs, ys):
            if x not in grouped_values:
                grouped_values[x] = []
            if y is None:
                y = np.nan
            grouped_values[x].append(y)
    # take the intersection
    xs = []
    ys = []
    for x in grouped_values:
        v = grouped_values[x]
        assert len(v) <= len(list_of_ys)
        if len(v) == len(list_of_ys):
            xs.append(x)
            ys.append(v)
    return np.asarray(xs), np.asarray(ys)


def smooth(x, extent):
    from scipy import signal
    if extent is not 0:
        return signal.savgol_filter(np.nan_to_num(x), extent,
                                    3)  # TODO: change the smoothing back to 101 when all the datapoints are in
    else:
        return np.nan_to_num(x)


class AxesWithPlots(object):
    def __init__(self, ax):
        self.ax = ax
        self.max_xs = []
        self.lines = []

    def add_std_plot(self, x, y, color, label, smoothen=51, alpha=1., std_alpha=0.3, frames_per_timestep=4, clip=None):
        x *= frames_per_timestep
        x /= 1e6
        if smoothen != 0:
            y = np.stack([smooth(y[:, i], extent=smoothen) for i in range(y.shape[1])], 1)
            if clip is not None:
                y = np.clip(y, clip[0], clip[1])
        mean_y = np.mean(y, 1)

        self.lines.extend(self.ax.plot(x, mean_y, color=color, label=label, alpha=alpha))
        if y.shape[1] > 1:
            std_y = np.std(y, 1, ddof=1)
            upper = mean_y + std_y / np.sqrt(y.shape[1])
            lower = mean_y - std_y / np.sqrt(y.shape[1])
            self.ax.fill_between(x, upper, lower, color=color, alpha=std_alpha, lw=0.)

        self.max_xs.append(x[-1])

    def finish_up(self, title, fontsize=8, xlim=400, tight_y=True):
        self.ax.set_xlim([0, min(xlim, max(*self.max_xs))])
        self.ax.set_title(title, fontsize=fontsize)
        self.ax.autoscale(enable=True, axis='y', tight=tight_y)


colors = {'idf': 'orange',
          'idf_3epochs': 'orange',
          'idf_6epochs': 'red',
          'vaesph': 'cornflowerblue',
          'rf': 'seagreen',
          'rf_6epochs': 'seagreen',
          'rf_3epochs': 'black',
          'extrew': 'gray',
          'ext': 'gray',
          'pix2pix': 'lightcoral'
          }
labels = {'idf': 'Inverse Dynamics\nfeatures',
          'idf_3epochs': 'Inverse Dynamics features 3 ep',
          'idf_6epochs': 'Inverse Dynamics features',
          'pix2pix': 'Pixels',
          'rf': 'Random CNN\nfeatures',
          'rf_3epochs': 'Random CNN features 3 ep',
          'rf_6epochs': 'Random CNN features',
          'extrew': 'Extrinsic rewards',
          'ext': 'Extrinsic rewards',
          'vaesph': 'VAE features'
          }


def generate_three_seed_graphs(three_seed_exps, y_series='eprew_recent', smoothen=51, alpha=1.0):
    num_envs = len(three_seed_exps.grouped_experiments)
    print(num_envs)
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, figsize=(12, 6))
    all_axes = []
    envs = [env for env in three_seed_exps.grouped_experiments if env not in ['Unity-Sparse']]
    envs = ['BeamRider', 'Breakout', 'MontezumaRevenge', 'Pong', 'Mario', 'Qbert', 'Riverraid', 'Seaquest', 'SpaceInvaders']
    # envs.remove('Mario')
    for env, ax in zip(envs, np.ravel(axes)):
        # import ipdb; ipdb.set_trace()
        ax.tick_params(labelsize=7, pad=0.001)
        ax_with_plots = AxesWithPlots(ax)
        all_axes.append(ax_with_plots)
        for method in three_seed_exps.grouped_experiments[env]:
            if method != 'extrew':
                print("generating graph ", env, method)
                xs, ys = three_seed_exps.grouped_experiments[env][method].get_xs_ys(y_series)
                ax_with_plots.add_std_plot(xs, ys, color=colors[method], label=labels[method], smoothen=smoothen,
                                           alpha=alpha)
        ax_with_plots.finish_up(title=env)
    axes[1,4].set_visible(False)
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.0, 0., 1, 1])
    # plt.tight_layout()
    fig.legend(handles=all_axes[0].lines, borderaxespad=0., fontsize=10, loc=(0.82,0.2), ncol=1)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center')
    fig.text(0.007, 0.5, 'Extrinsic Reward per Episode', va='center', rotation='vertical')
    save_filename = os.path.join(results_folder, 'three_seeds_{}.png'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()

def generate_three_seed_graphs_mario(three_seed_exps, y_series='eprew_recent', smoothen=51, alpha=1.0):
    num_envs = len(three_seed_exps.grouped_experiments)
    print(num_envs)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
    all_axes = []
    envs = [env for env in three_seed_exps.grouped_experiments if env in ['Mario']]
    for env, ax in zip(envs, np.ravel(axes)):
        # import ipdb; ipdb.set_trace()
        ax.tick_params(labelsize=4, pad=0.001)
        ax_with_plots = AxesWithPlots(ax)
        all_axes.append(ax_with_plots)
        for method in three_seed_exps.grouped_experiments[env]:
            if method != 'extrew':
                print("generating graph ", env, method)
                xs, ys = three_seed_exps.grouped_experiments[env][method].get_xs_ys(y_series)
                ax_with_plots.add_std_plot(xs, ys, color=colors[method], label=labels[method], smoothen=smoothen,
                                           alpha=alpha)
        ax_with_plots.finish_up(title=env)
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.01, 0., 1, 1])
    # plt.tight_layout()
    fig.legend(handles=all_axes[0].lines, borderaxespad=0., fontsize=10, loc='upper center', ncol=2)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center')
    fig.text(0.007, 0.5, 'Extrinsic Reward per Episode', va='center', rotation='vertical')
    save_filename = os.path.join(results_folder, 'three_seeds_mario_{}.png'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()


def generate_three_seed_graphs_unity(three_seed_exps, y_series='eprew_recent', smoothen=51, alpha=1.0):
    num_envs = len(three_seed_exps.grouped_experiments)
    print(num_envs)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
    all_axes = []
    envs = [env for env in three_seed_exps.grouped_experiments if env in ['Unity-Sparse']]
    for env, ax in zip(envs, np.ravel(axes)):
        # import ipdb; ipdb.set_trace()
        ax.tick_params(labelsize=12, pad=0.001)
        ax_with_plots = AxesWithPlots(ax)
        all_axes.append(ax_with_plots)
        for method in three_seed_exps.grouped_experiments[env]:
            print("generating graph ", env, method)
            xs, ys = three_seed_exps.grouped_experiments[env][method].get_xs_ys(y_series)
            ax_with_plots.add_std_plot(xs, ys, color=colors[method], label=labels[method], smoothen=smoothen,
                                       alpha=alpha, frames_per_timestep=1, clip=[0.,1.])
        ax_with_plots.finish_up(title=env, tight_y=False, fontsize=18)
        ax_with_plots.ax.set_xlim([0, 7])
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.01, 0.01, 1, 0.89])
    # plt.tight_layout()
    fig.legend(handles=all_axes[0].lines, borderaxespad=0., fontsize=12, loc='upper center', ncol=2)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center', fontsize=18)
    fig.text(0.007, 0.5, 'Extrinsic Reward per Episode', va='center', rotation='vertical', fontsize=18)
    save_filename = os.path.join(results_folder, 'three_seeds_unity_{}.png'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()

def generate_one_seed_graphs(one_seed_exps, y_series='eprew_recent'):
    num_envs = len(one_seed_exps.grouped_experiments)
    print(num_envs)
    fig, axes = plt.subplots(nrows=6, ncols=8, sharex=True)
    all_axes = []
    envs = [env for env in one_seed_exps.grouped_experiments if env not in ['Mario', 'Unity-Sparse']]
    for env, ax in zip(envs, np.ravel(axes)):
        # import ipdb; ipdb.set_trace()
        ax.tick_params(labelsize=4, pad=0.001)
        ax_with_plots = AxesWithPlots(ax)
        all_axes.append(ax_with_plots)
        for method in one_seed_exps.grouped_experiments[env]:
            if method != 'extrew' and '3epochs' not in method:
                print("generating graph ", env, method)
                xs, ys = one_seed_exps.grouped_experiments[env][method].get_xs_ys('eprew_recent')
                ax_with_plots.add_std_plot(xs, ys, color=colors[method], label=labels[method], smoothen=201)
        rand_agent_results_filename = os.path.join(results_folder, 'rand_agent', '{}_randagent_0'.format(env), 'log.txt')
        with open(rand_agent_results_filename, 'r') as f:
            rand_info = f.readlines()[-1]
            try:
                step, mean_r, best_r, worst_r = rand_info.split(':')
            except:
                print(env)
                step, mean_r, best_r = rand_info.split(':')
            mean_r = float(mean_r)
            best_r = float(best_r)
            # worst_r = float(worst_r)
        rand_ag_line = ax.axhline(mean_r, label='Random agent', color='blue')
        ax.set_title(env, fontsize=4)
        ax.autoscale(enable=True, axis='y', tight=False)
        ax.set_xlim([0, 400])
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.01, 0., 1, 0.925])
    # plt.tight_layout()
    fig.legend(handles=all_axes[0].lines+[rand_ag_line], borderaxespad=0., fontsize=10, loc='upper center', ncol=2)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center')
    fig.text(0.007, 0.5, 'Extrinsic Reward per Episode', va='center', rotation='vertical')
    save_filename = os.path.join(results_folder, 'one_seed_{}.png'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()

def generate_one_seed_aggregate_graph(one_seed_exps, y_series='eprew_recent'):
    num_envs = len(one_seed_exps.grouped_experiments)
    print(num_envs)
    all_better = {}
    # all_number = 0
    envs = [env for env in one_seed_exps.grouped_experiments if env not in ['Mario', 'Unity-Sparse']]
    all_better['comp'] = {'xs': [], 'ys': []}
    for env in envs:
        rand_agent_results_filename = os.path.join(results_folder, 'rand_agent', '{}_randagent_0'.format(env),
                                                   'log.txt')
        with open(rand_agent_results_filename, 'r') as f:
            rand_info = f.readlines()[-1]
            try:
                step, mean_r, best_r, worst_r = rand_info.split(':')
            except:
                print(env)
                step, mean_r, best_r = rand_info.split(':')
            mean_r = float(mean_r)
        for method in one_seed_exps.grouped_experiments[env]:
            if method != 'extrew' and '3epochs' not in method:
                print("generating graph ", env, method)
                xs, ys = one_seed_exps.grouped_experiments[env][method].get_xs_ys('eprew_recent')
                if method not in all_better:
                    all_better[method] = {'xs': [], 'ys': [], 'raw_ys': []}
                all_better[method]['xs'].append(xs)
                all_better[method]['ys'].append(ys > mean_r)
                all_better[method]['raw_ys'].append(ys.copy())
        all_better['comp']['xs'].append(all_better['rf_6epochs']['xs'][-1])
        idf_y = all_better['idf_6epochs']['raw_ys'][-1]
        rf_y = all_better['rf_6epochs']['raw_ys'][-1]
        cmn = min(idf_y.shape[0], rf_y.shape[0])
        all_better['comp']['ys'].append(rf_y[:cmn] > idf_y[:cmn])

    for method in all_better:
        # if 'rf' in method:
        n_common_pts = np.min([y.shape[0] for y in all_better[method]['ys']])
        print(n_common_pts)
        plot_y = np.asarray([y[:n_common_pts, 0] for y in all_better[method]['ys']])
        plt.plot(all_better[method]['xs'][0][:n_common_pts], smooth(np.mean(plot_y, 0), 51))
        plt.show()


# generate_one_seed_aggregate_graph(one_seed_exps)

def generate_mario_graphs(mario_exps, y_series='eprew_recent'):
    colors = {'idf_baseline': 'orange',
              'idf_transfer': 'lightcoral',
              'rf_baseline': 'seagreen',
              'rf_transfer': 'cornflowerblue', #"turquoise"
              # 'idf_fixed_baseline': 'yellow',
              # 'idf_fixed_transfer': 'green',
              }
    labels = {'idf_baseline': 'IDF scratch',
              'idf_transfer': 'IDF transfer',
              'rf_baseline': 'RF scratch',
              'rf_transfer': 'RF transfer',
              'idf_fixed_baseline': 'IDF fixed baseline',
              'idf_fixed_transfer': 'IDF fixed transfer',
              }

    gen_directions = [(1, 2), (1, 3)]
    # gen_directions = [(1, 3)]
    fig, axes = plt.subplots(nrows=1, ncols=len(gen_directions), sharex=True)
    all_axes = []
    exps = mario_exps.grouped_experiments['MarioGen']
    for gen_direction, ax in zip(gen_directions, np.ravel(axes)):
        ax.tick_params(labelsize=8, pad=0.001)
        ax_with_plots = AxesWithPlots(ax)
        all_axes.append(ax_with_plots)
        print("generating graph mariogen", '{}to{}'.format(*gen_direction))
        for method in ['idf', 'rf']:
            method_baseline = '{}to{}'.format(gen_direction[1], gen_direction[0]) + '_' + method
            if method_baseline in exps:
                xs_baseline, ys_baseline = exps[method_baseline].get_xs_ys(y_series)
                xs_baseline, ys_baseline = xs_baseline[10:xs_baseline.shape[0] // 2-10], ys_baseline[10:ys_baseline.shape[0] // 2-10]

                method_transfer = '{}to{}'.format(gen_direction[0], gen_direction[1]) + '_' + method
                xs_transfer, ys_transfer = exps[method_transfer].get_xs_ys(y_series)
                first_x = xs_transfer[xs_transfer.shape[0] // 2]
                xs_transfer, ys_transfer = xs_transfer[xs_transfer.shape[0] // 2+10:-10], ys_transfer[ys_transfer.shape[0] // 2+10:-10]
                xs_transfer -= first_x

                method_baseline_key = method + '_' + 'baseline'
                method_transfer_key = method + '_' + 'transfer'
                ax_with_plots.add_std_plot(xs_baseline, ys_baseline, color=colors[method_baseline_key],
                                           label=labels[method_baseline_key], smoothen=51, std_alpha=0.2)
                ax_with_plots.add_std_plot(xs_transfer, ys_transfer, color=colors[method_transfer_key],
                                           label=labels[method_transfer_key],
                                           smoothen=51, std_alpha=0.2)
        image = os.path.join(results_folder, 'mario-{}-1.gif').format(gen_direction[1])
        from PIL import Image
        image = Image.open(image)
        ax_with_plots.ax.imshow(image, extent=(0,40,0,150), aspect='auto')
        image = os.path.join(results_folder, 'mario-1-1.gif')
        image = Image.open(image)
        ax_with_plots.ax.imshow(image, extent=(0, 40, 220, 370), aspect='auto')
        # ax.text(20, 185, "TO", fontdict={"size": 5})
        ax.arrow(20, 212, 0, -30, width=5, facecolor='k')
        ax_with_plots.finish_up(title='World {} level 1 to world {} level 1'.format(*gen_direction), fontsize=10)
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=0., h_pad=0.3, rect=[0.01, 0., 1, 0.925])
    # plt.tight_layout()
    fig.legend(handles=all_axes[0].lines, borderaxespad=0., fontsize=10, loc='upper center', ncol=2)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center')
    fig.text(0.007, 0.5, 'Extrinsic Reward per Episode', va='center', rotation='vertical')
    save_filename = os.path.join(results_folder, 'mario_transfer_{}_non_fixed.pdf'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=1200)
    plt.close()


def generate_large_scale_graphs(large_scale_exps, y_series='eprew_recent', alpha=1.0, smoothen=51):
    colors = {'rf_small': 'g',
              'rf_large': 'r'
              }
    labels = {'rf_small': 'Batch of 128 environments',
              'rf_large': 'Batch of 2048 environments'
              }
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
    all_axes = []
    print(list(large_scale_exps.grouped_experiments['Mario'].keys()))
    small_scale_exp = large_scale_exps.grouped_experiments['Mario']['rf_small']
    large_scale_exp = large_scale_exps.grouped_experiments['Mario']['rf_large']
    ax = axes
    ax.tick_params(labelsize=8, pad=0.001)
    ax_with_plots = AxesWithPlots(ax)
    all_axes.append(ax_with_plots)
    grad_updates_per_optimization_step = 6*8
    xs, ys = small_scale_exp.get_xs_ys(y_series)
    y = np.mean(ys, 1)
    y = smooth(y, extent=smoothen)
    x = small_scale_exp.experiments[0].timeseries('n_updates')
    x = np.asarray(x) * grad_updates_per_optimization_step
    # x = np.asarray(x) / 3600
    ax.plot(x, y, color=colors['rf_small'], label=labels['rf_small'], alpha=alpha)
    x = large_scale_exp.experiments[0].timeseries('n_updates')
    x = np.asarray(x) * grad_updates_per_optimization_step
    # x = np.asarray(x) / 3600
    y = np.asarray(large_scale_exp.experiments[0].timeseries(y_series))
    y = smooth(y, extent=smoothen)
    ax.set_xlim((0.,200000.))
    ax.plot(x, y, color=colors['rf_large'], label=labels['rf_large'], alpha=alpha)

    # ax.set_xlim([0, min(xlim, max(*self.max_xs))])
    # ax.set_title('Scale in Mario', fontsize=10)
    ax.autoscale(enable=True, axis='y', tight=False)
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.01, 0.02, 1, 0.9])
    # plt.tight_layout()
    fig.legend(borderaxespad=0., fontsize=14, loc='upper center', ncol=1)
    fig.text(0.56, 0.01, 'Number of gradient updates', ha='center', fontsize=14)
    fig.text(0.007, 0.5, 'Extrinsic Reward per Episode', va='center', rotation='vertical', fontsize=14)
    save_filename = os.path.join(results_folder, 'large_scale_mario_{}.png'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()


def generate_unitytv_graphs(unity_exps, y_series='eprew_recent', smoothen=51, alpha=1.0):
    colors = {
        'rf-TV0': 'seagreen',
        'rf-TV2': 'cornflowerblue',
        'idf-TV0': 'orange',
        'idf-TV2': 'lightcoral'
    }
    labels = {
        'rf-TV0': 'RF with TV off',
        'rf-TV2': 'RF with TV on',
        'idf-TV0': 'IDF with TV off',
        'idf-TV2': 'IDF with TV on'
    }
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax = axes
    envs = [env for env in unity_exps.grouped_experiments]
    for env in envs:
        # import ipdb; ipdb.set_trace()
        ax.tick_params(labelsize=12, pad=0.001)
        ax_with_plots = AxesWithPlots(ax)
        for method in unity_exps.grouped_experiments[env]:
            print("generating graph ", env, method)
            xs, ys = unity_exps.grouped_experiments[env][method].get_xs_ys(y_series)
            method = method+'-'+env[-3:]
            ax_with_plots.add_std_plot(xs, ys, color=colors[method], label=labels[method], smoothen=smoothen,
                                           alpha=alpha, clip=[0.,1.])
        ax_with_plots.finish_up(title='Unity maze', fontsize=18)
    ax.set_xlim([0,8.5])
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.01, 0.01, 1, 0.925])
    # plt.tight_layout()
    fig.legend(borderaxespad=0., fontsize=12, loc='upper center', ncol=2)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center', fontsize=18)
    fig.text(0.007, 0.5, 'Extrinsic Reward per Episode', va='center', rotation='vertical', fontsize=18)
    save_filename = os.path.join(results_folder, 'three_seeds_unitytv_{}.png'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()

def generate_unitytv_graphs_compare_to_extrinsic(unity_exps, y_series='eprew_recent', smoothen=51, alpha=1.0):
    colors = {
        'rf-TV0': 'seagreen',
        'idf-TV0': 'orange',
    }
    labels = {
        'rf-TV0': 'Random CNN features',
        'idf-TV0': 'Inverse dynamics features',
    }
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax = axes
    envs = [env for env in unity_exps.grouped_experiments]
    for env in envs:
        if env[-1] == '0':
            # import ipdb; ipdb.set_trace()
            ax.tick_params(labelsize=12, pad=0.001)
            ax_with_plots = AxesWithPlots(ax)
            for method in unity_exps.grouped_experiments[env]:
                print("generating graph ", env, method)
                xs, ys = unity_exps.grouped_experiments[env][method].get_xs_ys(y_series)
                method = method+'-'+env[-3:]
                ax_with_plots.add_std_plot(xs, ys, color=colors[method], label=labels[method], smoothen=smoothen,
                                               alpha=alpha, clip=[0.,1.])
            ax_with_plots.finish_up(title='Unity maze', fontsize=18)
    ax.plot(np.zeros_like(ys)[:,0], color='gray', label='Extrinsic only')
    ax.set_xlim([0,8.5])
    ax.set_ylim([-0.01,1.01])
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.01, 0.01, 1, 0.925])
    # plt.tight_layout()
    fig.legend(borderaxespad=0., fontsize=12, loc='upper center', ncol=2)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center', fontsize=18)
    fig.text(0.007, 0.5, 'Extrinsic Reward per Episode', va='center', rotation='vertical', fontsize=18)
    save_filename = os.path.join(results_folder, 'three_seeds_unitytv_extrinsic_{}.png'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()


def generate_multipong_graphs(multipong_exps, y_series='eplen', smoothen=51, alpha=1.0):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax = axes
    envs = ['Multiplayer-Pong']
    for env in envs:
        # import ipdb; ipdb.set_trace()
        ax.tick_params(labelsize=12, pad=0.001)
        ax_with_plots = AxesWithPlots(ax)
        for method in multipong_exps.grouped_experiments[env]:
            print("generating graph ", env, method)
            xs, ys = multipong_exps.grouped_experiments[env][method].get_xs_ys(y_series)
            print(len(xs), len(ys))
            # ys = smooth(np.asarray(ys)[:,0], extent=smoothen)
            ax_with_plots.add_std_plot(xs, ys, color=colors[method], label="Pure curiosity (no-reward, infinite-horizon) exploration", smoothen=smoothen,
                                       alpha=alpha)
        image = os.path.join(results_folder, 'mplayerpong_weirdness.png')
        from PIL import Image
        image = Image.open(image)
        ax_with_plots.ax.imshow(image, extent=(215, 275, 5000, 8000), aspect='auto')
        image = os.path.join(results_folder, 'mplayerpong_normal.png')
        image = Image.open(image)
        ax_with_plots.ax.imshow(image, extent=(125, 185, 5000, 8000), aspect='auto')

        ax.set_title("Two player Pong", fontsize=18)
        ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_xlim([0,400])
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.01, 0.01, 1, 0.955])
    # plt.tight_layout()
    fig.legend(borderaxespad=0., fontsize=12, loc='upper center', ncol=2)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center', fontsize=18)
    fig.text(0.007, 0.5, 'Episode length', va='center', rotation='vertical', fontsize=18)
    save_filename = os.path.join(results_folder, 'multiplayer_pong_{}.png'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()


def generate_juggling_graphs(multipong_exps, y_series='eprew_recent', smoothen=51, alpha=1.0):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax = axes
    envs = ['Roboschool-Juggling']
    for env in envs:
        # import ipdb; ipdb.set_trace()
        ax.tick_params(labelsize=12, pad=0.001)
        ax_with_plots = AxesWithPlots(ax)
        for method in multipong_exps.grouped_experiments[env]:
            print("generating graph ", env, method)
            xs, ys = multipong_exps.grouped_experiments[env][method].get_xs_ys(y_series)
            print(len(xs), len(ys))
            # ys = smooth(np.asarray(ys)[:,0], extent=smoothen)
            ax_with_plots.add_std_plot(xs, ys, color=colors[method], label="Pure curiosity (no-reward, infinite-horizon) exploration", smoothen=smoothen,
                                       alpha=alpha, frames_per_timestep=1)

        ax.set_title("Juggling (Roboschool)", fontsize=18)
        ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_xlim([0,150])
    # fig.set_xlabel('Number of training steps (in millions)', fontsize=14)
    # fig.set_ylabel('Extrinsic Rewards per Episode', fontsize=14)
    plt.tight_layout(pad=1.5, w_pad=-0.3, h_pad=0.3, rect=[0.015, 0.01, 1, 0.955])
    # plt.tight_layout()
    fig.legend(borderaxespad=0., fontsize=12, loc='upper center', ncol=2)
    fig.text(0.5, 0.01, 'Frames (millions)', ha='center', fontsize=18)
    fig.text(0.007, 0.5, 'Ball bounces per episode', va='center', rotation='vertical', fontsize=18)
    save_filename = os.path.join(results_folder, 'juggling_{}.png'.format(y_series))
    print("saving ", save_filename)
    plt.savefig(save_filename, dpi=300)
    plt.close()

def main():
    # Code for Figure-2
    three_seed_exps = Experiments(os.path.join(results_folder, 'three_seed_main_envs'), reload_logs=False)
    generate_three_seed_graphs(three_seed_exps, 'eprew_recent')
    # # generate_three_seed_graphs(three_seed_exps, 'recent_best_ext_ret', smoothen=False, alpha=0.7)
    # generate_three_seed_graphs(three_seed_exps, 'best_ext_ret', smoothen=False)
    # generate_three_seed_graphs(three_seed_exps, 'eplen')

    # Code for Figure-8
    # one_seed_exps = Experiments(os.path.join(results_folder, 'one_seed_atari'), reload_logs=False)
    # generate_one_seed_graphs(one_seed_exps)
    # generate_one_seed_aggregate_graph(one_seed_exps)


if __name__ == '__main__':
    results_folder = os.path.expanduser('./')
    results_folder = '/..../experiments'
    main()
