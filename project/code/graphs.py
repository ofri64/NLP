import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

map = {'Baseline BiLSTM': 1,
       'SFMTL - Gender': 2,
       'SFMTL - Tense': 3,
       'SFMTL - Person': 4,
       'MFMTL': 5}

for graph_name in ('accuracy', 'ambigous', 'unseen'):
    file_path = './{0}.csv'.format(graph_name)

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

    plt.savefig('{0}.png'.format(graph_name))
    plt.show()


