# Versione del 2023-08-12 n.1

import pandas as pd
import numpy as np
from numpy import random
from sklearn.preprocessing import LabelEncoder

# librerie dei modelli usati
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier  # maximum likelihood estimation
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# metriche e suddivisione del modello
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# gestione del sistema
import time

# disablita i warning
import warnings

from xgboost import XGBClassifier, XGBRFClassifier

warnings.filterwarnings("ignore")

# Informazioni per il salvataggio in CSV
SUBJECT = 2
SW = 150  # in millisecondi
PREPROCESSING = "STD"
RIDUZIONE = "Kbest"
CANALI_USATI = "nessuno"
OVERLAP = 0
FEATURE_SELECTION_K = 1000


def classificatore(df, model):
    random.seed(0)

    start_time = time.time()

    # Numero di fold
    FOLDS = 5

    f1 = [None] * FOLDS
    accuracy = [None] * FOLDS
    precision = [None] * FOLDS
    recall = [None] * FOLDS
    confusionMatrix = [None] * FOLDS
    cols_idxs = [None] * FOLDS

    print("\nModello in uso: " + model.__class__.__name__ + "\n")

    print("Numero di fold usate: {}".format(FOLDS))

    for i in range(1, FOLDS + 1):
        df_train = df[df["id"] != i]
        df_test = df[df["id"] == i]
        print(
            "Dimensioni dataset di training: "
            + str(df_train.shape)
            + " | Dimensioni dataset di testing: "
            + str(df_test.shape)
        )

        # le sottoporzioni del dataset
        X_train = np.array(df_train.iloc[:, :-2])  # rimossa colonna gesto e ripetizione
        y_train = np.array(df_train["class"])  # colonna gesto
        X_test = np.array(df_test.iloc[:, :-2])  # rimossa colonna gesto e ripetizione
        y_test = np.array(df_test["class"])  # colonna gesto

        selector = SelectKBest(k=FEATURE_SELECTION_K)
        X_train = selector.fit_transform(X_train, y_train)
        # Get columns to keep and create new dataframe with those only
        cols_idxs[i-1] = selector.get_support(indices=True)
        X_test = X_test[:,cols_idxs[i-1]]

        print(
            "fold {} ".format(i)
            + "Dimensioni X_train: "
            + str(X_train.shape)
            + " | Dimensioni y_train: "
            + str(y_train.shape)
            + " | Dimensioni X_test: "
            + str(X_test.shape)
            + " | Dimensioni y_test: "
            + str(y_test.shape)
        )

        # adattamento valori delle classi per XGBClassifier
        if (
            model.__class__.__name__ == "XGBClassifier"
            or model.__class__.__name__ == "XGBRFClassifier"
        ):
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.fit_transform(y_test)

        # Training
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        f1[i-1] =  f1_score(y_test, y_pred, average="macro") * 100
        accuracy[i-1] = accuracy_score(y_test, y_pred)  * 100
        precision[i-1] =  precision_score(y_test, y_pred, average="macro") * 100
        recall[i-1] = recall_score(y_test, y_pred, average="macro") * 100
        confusionMatrix[i-1] = confusion_matrix(y_test, y_pred)

        # Results
        print("--------------------------------------------------------\n")
        print(
            "fold {} con K={}\n".format(i, FEATURE_SELECTION_K)
            + "Accuracy (fold {}): {:.2f}%\n".format(i, accuracy[i - 1])
            + " | F1 score (fold {}): {:.2f}%\n".format(i, f1[i - 1])
            + " | Precision (fold {}): {:.2f}%\n".format(i, precision[i - 1])
            + " | Recall (fold {}): {:.2f}%\n".format(i, recall[i - 1])
        )
        print("--------------------------------------------------------\n\n")

    accuracy_avg = np.mean(accuracy)
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1_avg = np.mean(f1)

    print("Accuracy avg: {:.2f}".format(accuracy_avg))
    print("Precision avg: {:.2f}".format(precision_avg))
    print("Recall avg: {:.2f}".format(recall_avg))
    print("F1 avg: {:.2f}".format(f1_avg))

    # Salvataggio delle informazioni in un csv
    result = pd.DataFrame(
        {
            "id_soggetto": [SUBJECT],
            "sw": [SW],
            "feature_usate": ["tutte"],
            "canali_esclusi": [CANALI_USATI],
            "overlap": [OVERLAP],
            "op_prepr": [PREPROCESSING],
            "op_riduz": [RIDUZIONE],
            "feature_selection": [FEATURE_SELECTION_K],
            "modello": [model.__class__.__name__],
            "iperparametri": [model.get_params()],
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1": [f1],
            "overall_accuracy": [accuracy_avg],
            "overall_precision": [precision_avg],
            "overall_recall": [recall_avg],
            "overall_f1": [f1_avg],
            "tempo_esecuzione": [float(time.time() - start_time)],
        }
    )

    result.to_csv(
        "Classifiers/Results/Results_one_subject_classification.csv",
        mode="a",
        header=False,
        index=False,
    )

    #salvataggio confusion matrix
    nomi_colonne_cf = [x for x in range(1,66)] #le colonne hanno dei nomi di numeri: 1,2,3,4,5,6,...,65

    # Crea 5 DataFrame dalle matrici
    df1 = pd.DataFrame(confusionMatrix[0], columns=nomi_colonne_cf)
    df2 = pd.DataFrame(confusionMatrix[1], columns=nomi_colonne_cf)
    df3 = pd.DataFrame(confusionMatrix[2], columns=nomi_colonne_cf)
    df4 = pd.DataFrame(confusionMatrix[3], columns=nomi_colonne_cf)
    df5 = pd.DataFrame(confusionMatrix[4], columns=nomi_colonne_cf)
    df_combined = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

    nomeSaveFile = "Classifiers/Results/CM/Results_ID_" + str(SUBJECT) + "_KB_" + str(FEATURE_SELECTION_K) + "_" + model.__class__.__name__ + "_CM.csv" #nome di salvataggio
    df_combined.to_csv(nomeSaveFile, mode='w', header=False, index=False)

    pd.DataFrame(cols_idxs).to_csv("Classifiers/Results/IndexesColumns/cols_idxs_id_" + str(SUBJECT) + ".csv", mode='w', header=False, index=False)

    # TODO: salvare la confusion matrix in un file csv

    print("Operazione conclusa in %s seconds \n" % (time.time() - start_time))


if __name__ == "__main__":
    # elenco dei modelli che vogliamo addestrarre
    models = [
        # classifictori visti dagli altri paper
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis(solver='svd')),
        ("Logistic Regression",LogisticRegression(C=0.01, solver='liblinear', penalty='l2', max_iter=2000, random_state=73, n_jobs=-1)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=73)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, criterion='log_loss', max_depth=8, max_features="sqrt", random_state=73, n_jobs=-1)),
        ("RidgeClassifier", RidgeClassifier(alpha=0.0002, random_state=73)),
        ("SVM", svm.SVC(C=50, gamma='scale', kernel='linear', random_state=73)),
        ("MLP", MLPClassifier(solver='adam', activation='identity', hidden_layer_sizes=[30,30,30,30,30, 30,30,30,30,30, 30,30,30], random_state=73)),
        ("LightGBM", LGBMClassifier(max_depth=15, num_leaves=25, learning_rate=0.05, random_state=73, n_jobs=-1, verbose=-1)),
        ("XGBClassifier", XGBClassifier(alpha=0, eta=0.12, gamma=0, max_delta_step=0, max_depth=6, min_child_weight=1.08, subsample=0.4, random_state=73, n_jobs=-1)),
        #("XGBRFClassifier", XGBRFClassifier(n_estimators=700, alpha=0, eta=0.12, gamma=0, max_delta_step=0, max_depth=6, min_child_weight=1.08, subsample=0.4, random_state=73, n_jobs=-1)),
    ]
    
    # modo banale per far girare tutti i classificatori: non un thread ciascuno ma in successione
    for i in [18 , 19 , 20]:
        SUBJECT = i
        print("Loading datasets of feature vectors of subject {}...".format(SUBJECT))
        df = pd.read_csv(
            #"D:\Tesi\Datasets\s"
            #+ str(SUBJECT) +
             "EMGdataset_"
            + str(SUBJECT)
            + ".feature_vectors_STD.csv"
        )
        print("Dataset loaded. Shape: " + str(df.shape))
        for m in models:
            classificatore(df, m[1])
