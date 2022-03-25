from deepface import *
from deepface.DeepFace import represent
import os
from os import path
import pandas as pd
import pickle
from tqdm import tqdm


def compare(img_path, db_path, model_name, distance_metric, enforce_detection = False, align = True, normalization = 'base'):
    if model_name == 'Ensemble':
        model_names = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
        metric_names = ['cosine', 'euclidean', 'euclidean_l2']
    elif model_name != 'Ensemble':
        model_names = []
        metric_names = []
        model_names.append(model_name)
        metric_names.append(distance_metric)

    file_name = "representations_%s.pkl" % model_name
    file_name = file_name.replace("-", "_").lower()

    if path.exists(db_path + "/" + file_name):
        f = open(db_path + '/' + file_name, 'rb')
        representations = pickle.load(f)

        print("There are ", len(representations), " representations found in ", file_name)

    else:  # create representation.pkl from scratch
        employees = []

        for r, d, f in os.walk(db_path):  # r=root, d=directories, f = files
            for file in f:
                if ('.jpg' in file.lower()) or ('.png' in file.lower()):
                    exact_path = r + "/" + file
                    employees.append(exact_path)

        representations = []
        pbar = tqdm(range(0, len(employees)), desc='Finding representations', disable=True)

        # for employee in employees:
        for employee in employees:
            instance = [employee]

            representation = represent(img_path=employee,
                                       model_name=model_name,
                                       enforce_detection=enforce_detection,
                                       align=align,
                                       normalization=normalization)

            instance.append(representation)

            # -------------------------------

            representations.append(instance)

        f = open(db_path + '/' + file_name, "wb")
        pickle.dump(representations, f)
        f.close()

        print("Representations stored in ", db_path, "/", file_name,
              " file. Please delete this file when you add new identities in your database.")

        df = pd.DataFrame(representations, columns=["identity", "%s_representation" % (model_name)])
        df_base = df.copy()
        resp_obj = []