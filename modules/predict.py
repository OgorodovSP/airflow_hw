# <YOUR_IMPORTS>
import glob
import json
import pandas as pd
import dill
import os
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    mod = os.listdir(f'{path}/data/models')
    mod.sort()

    with open(f'{path}/data/models/{mod[0]}', 'rb') as file:
        model = dill.load(file)
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for filename in glob.glob(f'{path}/data/test/*.json'):
        with open(filename) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df_tmp = pd.DataFrame(x)
            df_pred = pd.concat([df_pred, df_tmp], axis=0)

    df_pred.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
