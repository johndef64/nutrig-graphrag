#%%
import pandas as pd
file = 'log_gemma2_all-mpnet-base-v2_aws.log'
file = "log_qwen2.5_14b_dmis-lab_biobert-v1.1_aws.log"
df = pd.read_table(file, sep="[", header=None)
df

df.columns = ['time', 'message']
print(df['time'].head().to_csv())

# drop the lines in 'message' that not contains "processed successfully"
df = df[df['message'].str.contains("processed successfully")]
df = df.reset_index(drop=True)
#%%
df['time'] = df['time'].str.strip()
df['time'] = df['time'].str.replace(',', '.')
df['datetime'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f')

first_time = df['datetime'].iloc[0]
df['seconds'] = (df['datetime'] - first_time).dt.total_seconds()

"""
                      time  seconds
0  2025-06-03 14:23:35.429    0.000
1  2025-06-03 14:23:35.430    0.001
2  2025-06-03 14:23:35.430    0.001
3  2025-06-03 14:23:35.431    0.002
4  2025-06-03 14:23:35.431    0.002
"""
df
#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(df.index, df['seconds'], '-o', markersize=3)
plt.xlabel('Numero di documenti inseriti')
plt.ylabel('Secondi dal primo docuemnto')
plt.title('Curva dei secondi rispetto all\'indice')
plt.grid(True)
plt.show()

# %%
