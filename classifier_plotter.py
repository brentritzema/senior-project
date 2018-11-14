import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from fire import Fire

def main(results_csv):
    ###############################################
    #Plot the results.

    df_results = pd.read_csv(results_csv)

    f1_means = df_results['f1_score']
    f1_stds = df_results['sd']

    _, ax = plt.subplots(figsize=(8, max(len(f1_means), 1.5)))
    y_pos = np.arange(len(f1_means))
    bars = ax.barh(y_pos, f1_means, height=0.5, align='center', xerr=f1_stds)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([type(x).__name__ for x in CLASSIFIERS])
    plt.gca().invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('Stance XVal F1-Weighted Score')
    ax.set_title('Classifiers for the SemEval 2016 Task A Dataset')

    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.text(rect.get_width() + 0.01, rect.get_y() + rect.get_height()/5.0,
                    '{:.3f}'.format(width),
                    ha='left', va='center')
    autolabel(bars)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    Fire(main)