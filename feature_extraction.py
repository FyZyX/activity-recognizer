from visualize import visualize
import numpy as np
import pandas as pd
import os

data_dir = 'data'


def max_amplitude(z):
    """
    Extracts the dominant amplitude via Fourier analysis
    :param z: time series
    :return: dominant magnitude
    """
    return max(map(lambda x: np.absolute(x), np.fft.fft(z)))


def amplitude_fraction(z):
    """
    Method to compare two frequency amplitudes. Prints a percentage of the dominant magnitude
    """
    amps = sorted(map(lambda x: np.absolute(x), np.fft.fft(z)), reverse=True)

    def amp_to_percent(amp):
        return round(100 * amp / amps[0], 2)

    print('{}% of f_0'.format(max([amp_to_percent(amp) for amp in amps[1:10]])))


def create_feature_vector(activity, df):
    """
    Convenience method to wrap features
    """
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
    """
    Extracts feature vectors from raw data
    :return: DataFrame with extracted features
    """
    raw_dir = os.path.join(data_dir, 'raw')
    extracted_features = []

    # loop through the files for each participant
    for data_file in os.listdir(raw_dir):
        # print(data_file)
        df = pd.read_csv(os.path.join(raw_dir, data_file), index_col=0, names=['x', 'y', 'z', 'activity'])
        # group data points by activity
        activities = df.groupby('activity')

        for activity, activity_df in activities:
            # ignore activity 0, which is a test point
            if activity == 0:
                continue

            activity_df = activity_df.drop('activity', axis=1)

            # summarize data
            print("Activity", activity)
            print(activity_df.describe())

            # produce acceleration graphs
            # visualize(activity_df)

            # determine if other amplitudes should be included in feature vector
            # amplitude_fraction(activity_df.x)
            # amplitude_fraction(activity_df.y)
            # amplitude_fraction(activity_df.z)

            extracted_features.append(create_feature_vector(activity, activity_df))

    return pd.DataFrame(extracted_features)


if __name__ == '__main__':
    feature_vectors = extract()

    # write to file
    # feature_vectors.to_csv(os.path.join(data_dir, 'data.csv'))
