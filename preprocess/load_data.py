import pandas as pd
import os


def read_data_from_folders(data_path: str) -> list:
    '''
    If the data is located in your_working_dir/data and zips are extracted into
    this folder the function should automatically load all data in dfs. See map_folder dict for reference.
    :param data_path:
    :return:
    '''
    map_folder = {'sample': pd.DataFrame(),
                  'train': pd.DataFrame(),
                  'val': pd.DataFrame()}

    for folder in map_folder.keys():
        data_subfolder_path: str = os.path.join(data_path, folder)
        df_list: list = []
        for filename in os.listdir(data_subfolder_path):
            if filename.endswith(('.json', '.jsonl')):
                df_list.append(pd.read_json(os.path.join(data_subfolder_path, filename), lines=True))
        map_folder[folder] = pd.concat(df_list)

    return [map_folder['sample'], map_folder['train'], map_folder['val']]

