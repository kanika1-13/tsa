import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --------- Set parameters for year 2025 ---------
start_date = datetime(2025, 1, 1)
n_timestamps = 365 * 24 * 60  # Number of minutes in one year (non-leap)
freq = '1min'  # 1-minute intervals

# --------- Generate timestamps and DataFrame ---------
date_rng = pd.date_range(start=start_date, periods=n_timestamps, freq=freq)
df = pd.DataFrame(date_rng, columns=['timestamp'])

# --------- Helper functions ---------
def seasonal_pattern(timestamps, period, amplitude, phase_shift=0):
    return amplitude * np.sin(2 * np.pi * (timestamps / period + phase_shift))

def daily_cycle(timestamps, amplitude):
    day_fraction = (timestamps % 1440) / 1440
    return amplitude * -np.cos(2 * np.pi * day_fraction)

# --------- Generate features ---------
timestamps = np.arange(n_timestamps)
yearly_pattern = seasonal_pattern(timestamps, 525600, 10)  # 525600 = minutes in 1 year
daily_pattern = daily_cycle(timestamps, 5)
temp_trend = np.linspace(20, 22, n_timestamps)
temp_noise = np.random.normal(0, .1, n_timestamps)
df['temperature'] = temp_trend + yearly_pattern + daily_pattern + temp_noise

pressure_base = 100 + seasonal_pattern(timestamps, 525600, 1)
pressure_noise = np.random.normal(0, 0.01, n_timestamps)
df['pressure'] = pressure_base + pressure_noise

power_yearly = seasonal_pattern(timestamps, 525600, 10)
power_weekly = seasonal_pattern(timestamps, 10080, 5)  # 10080 = minutes in a week
power_daily = daily_cycle(timestamps, 15)
power_noise = np.random.normal(0, .2, n_timestamps)
df['power'] = 100 + power_yearly + power_weekly + power_daily + power_noise

# --------- Add anomalies ---------
def add_point_anomalies(anomaly_category, series, y, frequency=0.0025, anomaly_mean=2, anomaly_std=.25):
    anomaly_mask = np.random.random(len(series)) < frequency
    y[anomaly_mask] = anomaly_category
    series[anomaly_mask] += np.power(-1, np.random.random(size=(np.sum(anomaly_mask),)) > .5) * \
                            np.random.lognormal(mean=anomaly_mean, sigma=anomaly_std, size=(np.sum(anomaly_mask),))
    return series

df['anomaly'] = [0 for _ in range(n_timestamps)]
df['temperature'] = add_point_anomalies(1, df['temperature'], df['anomaly'], anomaly_mean=.5)
df['pressure'] = add_point_anomalies(2, df['pressure'], df['anomaly'], anomaly_mean=-2, anomaly_std=.08)
df['power'] = add_point_anomalies(3, df['power'], df['anomaly'], anomaly_mean=1.5)

# --------- Save data to CSV ---------
print("Saving data to CSV...")
df.to_csv('data_2025.csv', index=False)
print("Data saved successfully.")

# --------- Calculate Mean and Standard Deviation ---------
means = df[['temperature', 'pressure', 'power']].mean()
stds = df[['temperature', 'pressure', 'power']].std()

# Save the means and standard deviations as .npy files
np.save("means.npy", means.to_numpy())
np.save("stds.npy", stds.to_numpy())

print("Means and standard deviations saved successfully.")

# --------- Plot a sample of the data ---------
# --------- Plot a sample of the data ---------
sample_size = 7 * 24 * 60  # 1 week
fig, axs = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
variables = ['temperature', 'pressure', 'power']
titles = ['Temperature (°C)', 'Pressure (kPa)', 'Power Consumption (kW)']

for i, (var, title) in enumerate(zip(variables, titles)):
    anomalous_data = df[df["anomaly"] == (i + 1)]
    axs[i].plot(df['timestamp'][:sample_size], df[var][:sample_size], label=title)
    axs[i].set_title(title, fontsize=14, pad=12)
    axs[i].set_ylabel('Value')
    axs[i].plot(anomalous_data["timestamp"], anomalous_data[var], marker='o', ms=4, linestyle='', color='red', label="Anomaly")
    axs[i].legend(loc='upper right')

axs[-1].set_xlabel('Timestamp')
plt.tight_layout(pad=5)  # Increase padding to avoid overlapping
plt.show()
