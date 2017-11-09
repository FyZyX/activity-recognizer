import numpy as np
import pandas as pd
import os

data_dir = 'data'


def max_amplitude(z):
    return max(map(lambda x: np.absolute(x), np.fft.fft(z)))


def extract():
    extracted_features = []

    for data_file in filter(lambda x: x.endswith('.csv'), os.listdir(os.path.join(data_dir,'raw'))):
        print(data_file)
        df = pd.read_csv(os.path.join(data_dir, data_file), index_col=0, names=['x', 'y', 'z', 'activity'])
        activities = df.groupby('activity')

        for activity, activity_df in activities:
            if activity == 0:
                continue

            freq_x = max_amplitude(activity_df.x)
            freq_y = max_amplitude(activity_df.y)
            freq_z = max_amplitude(activity_df.z)
            features = {"x_mean": activity_df.x.mean(),
                        "y_mean": activity_df.y.mean(),
                        "z_mean": activity_df.z.mean(),
                        "x_std": activity_df.x.std(),
                        "y_std": activity_df.y.std(),
                        "z_std": activity_df.z.std(),
                        "x_freq": freq_x,
                        "y_freq": freq_y,
                        "z_freq": freq_z,
                        "activity": activity}

            extracted_features.append(features)

    return pd.DataFrame(extracted_features)


extract().to_csv(os.path.join(data_dir, 'data.csv'))
