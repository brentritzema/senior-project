from fire import Fire

import pandas as pd

def main(results_fp):
    df_results = pd.DataFrame().from_csv(results_fp)
    print(df_results)

if __name__ == '__main__':
    Fire(main)
