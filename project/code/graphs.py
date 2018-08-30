import pandas as pd
import matplotlib.pyplot as plt

graph_name = 'unseen'
file_path = './{0}.csv'.format(graph_name)

df = pd.read_csv(file_path).transpose()

plt.figure()
plt.tight_layout()

df.plot(grid=True, style='.-')

languages = df.index.values
plt.xticks(range(len(languages)), languages)

plt.ylabel('Accuracy')
plt.legend(['m1', 'm2', 'm3', 'm4'], loc='best')

plt.savefig('{0}.png'.format(graph_name))
plt.show()


