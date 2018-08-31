import pandas as pd
import matplotlib.pyplot as plt

file_path = './results/internal_external.csv'

df = pd.read_csv(file_path).transpose()
models = list(df.iloc[0].values)
languages = df.index.values[1:]

df.iloc[0] = pd.to_numeric(df.iloc[0], errors='coerce')
df.drop(df.index[0], inplace=True)

plt.figure()
plt.tight_layout()

df.plot(grid=True, style='.-')

plt.xticks(range(len(languages)), languages)

plt.ylabel('Accuracy')
plt.legend(models, loc='best')

plt.savefig('./images/internal_external.png')
plt.show()


