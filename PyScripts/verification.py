import sys
import os
from os import path
from datetime import datetime
from deepface import DeepFace
from deepface.commons import functions
import numpy as np
import pandas as pd
import cv2
import time
from DeepFaceAPI import *


def save_result(image_path, result_distance, result_score, result_knn):
    # (width, height)

    result_distance_pred = result_distance[0][0]
    result_score_pred = result_score[0][0]
    result_knn_pred = result_knn[0][0]

    img = cv2.imread(image_path)
    cv2.resize(img, (100, 120))

    result_dir = 'C:\\Dev\\Github\\TiagoSanti\\deep-face\\Results\\'
    try:
        os.mkdir(result_dir)
    except FileExistsError:
        pass

    datet = datetime.now().strftime('%S_%f')
    results = f'd.{result_distance_pred}-s.{result_score_pred}-k.{result_knn_pred}_'
    file_name = result_dir + results + datet + '.jpg'
    cv2.imwrite(file_name, img)
    return file_name


try:
    database_path = sys.argv[1]
    temp_path = sys.argv[2]
except IndexError:
    database_path = r'C:\Dev\Github\TiagoSanti\deep-face\Database'
    temp_path = r'C:\Dev\Github\TiagoSanti\deep-face\Temp'

model_name = 'ArcFace'
metric = 'cosine'

while True:
    start = time.time()

    temp_files = os.listdir(temp_path)

    if len(temp_files) > 0:
        for temp_file in temp_files:
            temp_file_path = temp_path + '\\' + temp_file

            temp_img = cv2.imread(temp_file_path)

            result = DeepFace.find(temp_img,
                                   db_path=database_path,
                                   model_name=model_name,
                                   distance_metric=metric,
                                   detector_backend='mtcnn',
                                   enforce_detection=False)

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

            print(result)
            print()
            sys.stdout.flush()

            distance_mean = []
            score_mean = []

            # mean distances and scores
            for id_name in unique_id:
                id_distances = result[result['identity'] == id_name][f'{model_name}_{metric}'].values
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

            result_distance = df.sort_values(by=['distance_mean'])[:1]
            result_score = df.sort_values(by=['score_mean'], ascending=False)[:1]
            result_knn = df.sort_values(by=['knn'], ascending=False)[:1]

            result_file_name = save_result(temp_file_path,
                                           result_distance.values.tolist(),
                                           result_score.values.tolist(),
                                           result_knn.values.tolist())

            print('\nRESULT FILE:', result_file_name.split('\\')[-1])
            print('Result by distance mean:')
            print(result_distance)

            print('\nResult by score mean:')
            print(result_score)

            print('\nResult by k_neighbors:')
            print(result_knn)

            print(f'\n{(time.time() - start):.2f} seconds to verify\n')
            sys.stdout.flush()

            os.remove(temp_file_path)

    time.sleep(5)
