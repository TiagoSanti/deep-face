import sys
import os
from deepface import DeepFace
import numpy as np
import pandas as pd
import time


def delete_temp_image(image_path):
    os.remove(image_path)


database_path = sys.argv[1]
temp_path = sys.argv[2]

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
                                   model_name='VGG-Face',
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
                id = path1.split('\\')[-1]
                sys.stdout.flush()

                identity.append(id)

                if id not in unique_id:
                    unique_id.append(id)

            result['identity'] = identity

            # scores
            score = []
            for i in result.values:
                score.append(1 / np.square(i[1]))

            result['score'] = score

            distance_mean = []
            score_mean = []

            # mean distances and scores
            for id in unique_id:
                id_distances = result[result['identity'] == id]['VGG-Face_cosine'].values
                id_scores = result[result['identity'] == id]['score'].values

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
            for id in ids_in_neighborhood:
                id_count.append(neighborhood_slice[neighborhood_slice['identity'] == id].shape[0])

            df['knn'] = [0] * df.shape[0]
            knn_column = df.columns.get_loc('knn')

            for id, value in zip(ids_in_neighborhood, id_count):
                id_index = df.index[df['id'] == id].tolist()[0]
                df.iat[id_index, knn_column] = value

            print(df)
            sys.stdout.flush()
            # Printing results
            '''
            print('Result by distance mean:')
            print(df.sort_values(by=['distance_mean'])[:1])

            print('\nResult by score mean:')
            print(df.sort_values(by=['score_mean'], ascending=False)[:1])

            print('\nResult by k_neighbors:')
            print(df.sort_values(by=['knn'], ascending=False)[:1])

            print(f'\n{(time.time()-start):.2f} seconds to verify\n')
            sys.stdout.flush()
            '''
            delete_temp_image(temp_file_path)

    time.sleep(3)
