import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

def load_raw_data(s_dir='./'):
    """expects directory to have multiple samples%d.npy"""

    sample_files = []
    count = 1
    def get_filename(s_dir, count):
        return os.path.join(s_dir, f"samples{count}.npy")

    current_filename = get_filename(s_dir, count)
    while os.path.exists(current_filename):
        sample_files.append(current_filename)
        count += 1
        current_filename = get_filename(s_dir, count)

    return [np.load(sf, allow_pickle=True) for sf in sample_files]


def preprocess_data(raw_data, increase_noise=False):
    """
    Formats PHY data, normalises
    """
    new_data = []
    for raw_region in raw_data:
        processed_region = raw_region[:, :-1].copy()
        phy_r = raw_region[:, -1].copy()
        phy = np.concatenate([abs(s.reshape((1, -1))) for s in phy_r], axis=0)
        if increase_noise:
            phy += 1e-3 * np.random.random(size=phy.shape)
        processed_region = np.concatenate((processed_region, phy), axis=1).astype(float)
        new_data.append(processed_region)

    n_regions = len(new_data)
    ds_lengths = [len(d) for d in new_data]
    normalised = MinMaxScaler().fit_transform(np.concatenate(new_data, axis=0))

    end_markers = [0] + [sum(ds_lengths[: i+1]) for i in range(n_regions)]
    separated = [normalised[end_markers[i] : end_markers[i+1]] for i in range(n_regions)]
    return separated


def load_data_from_regions(s_dir='./'):
    return preprocess_data(load_raw_data(s_dir))


def load_data():
    return np.load("wireless_ds.npy", allow_pickle=True)


def store_data(dataset):
    # 80% training, 20% test
    ratio = 0.8
    splits = [int(len(ds) * ratio) for ds in dataset]
    train_sets = [ds[:splits[i]] for i, ds in enumerate(dataset)]
    test_sets = [ds[splits[i]:] for i, ds in enumerate(dataset)]
    ds = np.array((label(train_sets), label(test_sets)), dtype=object)
    np.save("wireless_ds.npy", ds)
    return ds


def get_labels(ds_list):
    return np.concatenate([i * np.ones((len(ds), 1)) for i, ds in enumerate(ds_list)])


def label(ds_list):
    return np.concatenate(ds_list), get_labels(ds_list)


def plot_examples(dataset):
    SMALL_SIZE = 12
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    f = plt.figure()
    n_features = 20
    for i, ds in enumerate(dataset):
        #plt.plot(np.mean(ds[:, :n_features], axis=0), label=f"Region {i}")
        plt.boxplot([ds[:, ii] for ii in range(n_features)])
        if i == 0:
            break

    #plt.legend()
    plt.xlabel("Feature Index")
    plt.ylabel("Normalised Feature Value")

    plt.savefig("sample_means_box.pdf", bbox_inches="tight")


def plot_overlap(train_sets):
    f = plt.figure()
    n_features = 20

    for f_i in range(n_features):
        data = [ds[:, f_i] for ds in train_sets]
        plt.boxplot(data)

        plt.legend()
        plt.xlabel("DS Index")
        plt.ylabel("Normalised Feature Values")
        #plt.savefig("sample_means_box.pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    dataset = load_data_from_regions()
    plot_examples(dataset)
    #plot_overlap(dataset)
    #ds = store_data(dataset)
