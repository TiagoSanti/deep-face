import sys
import os
from deepface import DeepFace
import numpy as np
import pandas as pd
import time

try:
    database_path = sys.argv[1]
    temp_path = sys.argv[2]
except:
    database_path = r'C:\Dev\Github\TiagoSanti\deep-face\Database'
    temp_path = r'C:\Dev\Github\TiagoSanti\deep-face\Temp'

while True:
    start = time.time()

    temp_files = os.listdir(temp_path)

    if len(temp_files) > 0:
        result = None
        print('begin verification')
        for temp_file in temp_files:
            temp_file_path = temp_path + '\\' + temp_file
            result = DeepFace.find(temp_file_path,
                                   db_path=database_path,
                                   model_name='ArcFace',
                                   distance_metric='cosine',
                                   detector_backend='mtcnn',
                                   enforce_detection=False)

        if result is not None:
            # files paths
            paths = result['identity'].values.tolist()
            identity = []
            unique_id = []

            # getting people names
            for path in paths:
                path1 = path.split('/')[-2]
                id_name = path1.split('\\')[-1]
                sys.stdout.flush()

                identity.append(id_name)

                if id_name not in unique_id:
                    unique_id.append(id_name)

            result['identity'] = identity

            # scores
            score = []
            for i in result.values:
                score.append(1 / np.square(i[1]))

            result['score'] = score

            distance_mean = []
            score_mean = []

            # mean distances and scores
            for id_name in unique_id:
                id_distances = result[result['identity'] == id_name]['ArcFace_cosine'].values
                id_scores = result[result['identity'] == id_name]['score'].values

                id_distances_lenght = len(id_distances)
                id_scores_lenght = len(id_scores)

                id_distances_mean = sum(id_distances) / id_distances_lenght
                id_scores_mean = sum(id_scores) / id_scores_lenght

                distance_mean.append(id_distances_mean)
                score_mean.append(id_scores_mean)

            # final dataframe
            df = pd.DataFrame({
                'id': unique_id,
                'distance_mean': distance_mean,
                'score_mean': score_mean
            })

            # knn visualization
            k_neighbors = 4
            neighborhood_slice = result[:k_neighbors]
            ids_in_neighborhood = neighborhood_slice['identity'].unique()

            id_count = []
            for id_name in ids_in_neighborhood:
                id_count.append(neighborhood_slice[neighborhood_slice['identity'] == id_name].shape[0])

            df['knn'] = [0] * df.shape[0]
            knn_column = df.columns.get_loc('knn')

            for id_name, value in zip(ids_in_neighborhood, id_count):
                id_index = df.index[df['id'] == id_name].tolist()[0]
                df.iat[id_index, knn_column] = value

            print(df)
            sys.stdout.flush()
            # Printing results

            print('\nResult by distance mean:')
            print(df.sort_values(by=['distance_mean'])[:1])

            print('\nResult by score mean:')
            print(df.sort_values(by=['score_mean'], ascending=False)[:1])

            print('\nResult by k_neighbors:')
            print(df.sort_values(by=['knn'], ascending=False)[:1])

            print(f'\n{(time.time() - start):.2f} seconds to verify\n')
            sys.stdout.flush()

            os.remove(temp_file_path)

    time.sleep(10)
