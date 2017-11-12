from visualize import visualize
import numpy as np
import pandas as pd
import os
from random import choices

data_dir = 'data'


def max_amplitude(z):
    return max(map(lambda x: np.absolute(x), np.fft.fft(z)))


def create_feature_vector(activity, df):
    return {"x_mean": df.x.mean(),
            "y_mean": df.y.mean(),
            "z_mean": df.z.mean(),
            "x_std": df.x.std(),
            "y_std": df.y.std(),
            "z_std": df.z.std(),
            "x_freq": max_amplitude(df.x),
            "y_freq": max_amplitude(df.y),
            "z_freq": max_amplitude(df.z),
            "activity": activity}


def extract():
    raw_dir = os.path.join(data_dir, 'raw')
    extracted_features = []

    for data_file in [os.listdir(raw_dir)[0]]:  # choices(, k=1):
        print(data_file)
        df = pd.read_csv(os.path.join(raw_dir, data_file), index_col=0, names=['x', 'y', 'z', 'activity'])
        activities = df.groupby('activity')

        for activity, activity_df in activities:
            if activity == 0:
                continue

            activity_df = activity_df.drop('activity', axis=1)

            visualize(activity_df)

            # extracted_features.append(create_feature_vector(activity, activity_df))

    return pd.DataFrame(extracted_features)


feature_vectors = extract()

# write to file
# feature_vectors.to_csv(os.path.join(data_dir, 'data.csv'))
