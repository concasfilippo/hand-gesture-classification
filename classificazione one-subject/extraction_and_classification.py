# Versione del 2023-09-22

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

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector


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

from feature_extraction import extraction

warnings.filterwarnings("ignore")

# Informazioni per il salvataggio in CSV
SUBJECT = 2
SW = 150  # in millisecondi
PREPROCESSING = "STD"
RIDUZIONE = "Kbest"
CANALI_USATI = "nessuno"
OVERLAP = 0
FEATURE_SELECTION_K = 1000


def pre_processing_e_selezione_features(df):
    print("pre_processing")
    random.seed(0)

    preprocessing_total_time = time.time()

    # Numero di fold
    FOLDS = 5

    cols_idxs = [None] * FOLDS
    selection_time_stamps = [None] * FOLDS

    print("Selezione delle migliori {} features in corso...".format(FEATURE_SELECTION_K))
    for i in range(1, FOLDS + 1):
        df_train = df[df["id"] != i]
        df_test = df[df["id"] == i]

        # le sottoporzioni del dataset
        X_train = np.array(df_train.iloc[:, :-2])  # rimossa colonna gesto e ripetizione
        y_train = np.array(df_train["class"])  # colonna gesto
        X_test = np.array(df_test.iloc[:, :-2])  # rimossa colonna gesto e ripetizione
        y_test = np.array(df_test["class"])  # colonna gesto

        #selezione delle k feature migliori
        selection_timer = time.time()
        ###versione con KBEST
        selector = SelectKBest(k=FEATURE_SELECTION_K)
        X_train = selector.fit_transform(X_train, y_train)

        ###versione con forward feature selection
        models_ = LinearDiscriminantAnalysis(solver='svd')
        selector = SequentialFeatureSelector(models_, n_features_to_select=3, scoring='f1_weighted', n_jobs=-1, cv=5)
        
        X_train = selector.fit_transform(X_train, y_train)

        # Get columns to keep and create new dataframe with those only
        cols_idxs[i-1] = selector.get_support(indices=True)
        X_test = X_test[:,cols_idxs[i-1]]
        selection_time_stamps[i-1] = time.time() - selection_timer

    print("Selezione completata in {:.2f} secondi".format(sum(selection_time_stamps)))

    #costruisco un insieme con gli indici delle feature comuni selezionate nelle 5 fold
    lista_nomi_features = pd.read_csv('new_labels.csv', header=None)
    feature_id_comuni = cols_idxs[0]
    #faccio l'intersezioni per ottenerle le feature comuni a tutte le fold
    for i in range(1,5):
        feature_id_comuni = list(set(feature_id_comuni) & set(cols_idxs[i]))

    #ottengo le label delle feature comuni
    nomi_features_comuni = lista_nomi_features.iloc[:,feature_id_comuni]
    print("Numero di feature comuni: {}".format(nomi_features_comuni.shape[1])) 

    #estrazione delle feature comuni
    extraction_timer = time.time()
    print("Estrazione delle feature comuni in corso...")
    df_extraction, standardizzazione_time = extraction(nomi_features_comuni, SUBJECT) #funzione nostra
    extraction_time = time.time() - extraction_timer
    print("Estrazione completata in {:.2f} secondi".format(extraction_time))


    return df_extraction, nomi_features_comuni, extraction_time, selection_time_stamps, standardizzazione_time, time.time() - preprocessing_total_time



def classificatore(model, elements):
    df_extraction, nomi_features_comuni, extraction_time, selection_time_stamps, standardizzazione_time, preprocessing_total_time = elements

    start_time = time.time()
    
    # Numero di fold
    FOLDS = 5

    f1 = [None] * FOLDS
    accuracy = [None] * FOLDS
    precision = [None] * FOLDS
    recall = [None] * FOLDS
    confusionMatrix = [None] * FOLDS
    # cols_idxs = [None] * FOLDS
    training_time_stamps = [None] * FOLDS
    prediction_time_stamps = [None] * FOLDS

    print("\nInizio Classificazione\nModello in uso: " + model.__class__.__name__ + "\n")
    for i in range(1, FOLDS + 1):
        # print("df_extraction \n")
        # print(df_extraction)
        df_train = df_extraction[df_extraction["idx"] != i]
        df_test = df_extraction[df_extraction["idx"] == i]
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
        training_timer = time.time()
        model.fit(X_train, y_train)
        training_time_stamps[i-1] = time.time() - training_timer

        # Prediction
        prediction_timer = time.time()
        y_pred = model.predict(X_test)
        prediction_time_stamps[i-1] = time.time() - prediction_timer

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

    # Salvataggio dei risultati della classificazione in un csv
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
            "tempo_estrazione": [extraction_time],
            "tempo_standardizzazione": [standardizzazione_time],
            "tempo_selezione_totale": [selection_time_stamps],
            "tempi_training_totale": [training_time_stamps],
            "tempi_predizioni_totale": [prediction_time_stamps],
            "tempo_esecuzione_totale": [float(time.time() - start_time) + float(preprocessing_total_time)],
        }
    )

    result.to_csv(
        "Classifiers/Results/Results_one_subject_extraction_and_classification.csv",
        mode="a",
        header=False,
        index=False,
    )

    #salvataggio confusion matrix
    nomi_colonne_cf = [x for x in range(1,66)] #le colonne hanno dei nomi di numeri: 1,2,3,4,5,6,...,65

    #Crea 5 DataFrame dalle matrici
    df_cm1 = pd.DataFrame(confusionMatrix[0], columns=nomi_colonne_cf)
    df_cm2 = pd.DataFrame(confusionMatrix[1], columns=nomi_colonne_cf)
    df_cm3 = pd.DataFrame(confusionMatrix[2], columns=nomi_colonne_cf)
    df_cm4 = pd.DataFrame(confusionMatrix[3], columns=nomi_colonne_cf)
    df_cm5 = pd.DataFrame(confusionMatrix[4], columns=nomi_colonne_cf)
    df_cm_combined = pd.concat([df_cm1, df_cm2, df_cm3, df_cm4, df_cm5], ignore_index=True)

    nomeSaveFile = "Classifiers/Results/CM/Results_ID_" + str(SUBJECT) + "_KB_" + str(FEATURE_SELECTION_K) + "_" + model.__class__.__name__ + "_CM.csv" #nome di salvataggio
    df_cm_combined.to_csv(nomeSaveFile, mode='w', header=False, index=False)

    pd.DataFrame(nomi_features_comuni).to_csv("Classifiers/Results/IndexesColumns/cols_idxs_id_" + str(SUBJECT) + "_K_" + str(FEATURE_SELECTION_K) + ".csv", mode='w', header=False, index=False)
    ###TODO da rivedere, potrebbe essere sovrascritto lasciando liste vuote

    #salvataggio delle features comuni
    labels_comuni_with_extra_values = nomi_features_comuni.copy() #forse possiamo toglierla
    labels_comuni_with_extra_values["selecton_time_1"] = selection_time_stamps[0]
    labels_comuni_with_extra_values["selecton_time_2"] = selection_time_stamps[1]
    labels_comuni_with_extra_values["selecton_time_3"] = selection_time_stamps[2]
    labels_comuni_with_extra_values["selecton_time_4"] = selection_time_stamps[3]
    labels_comuni_with_extra_values["selecton_time_5"] = selection_time_stamps[4]
    labels_comuni_with_extra_values.to_csv("Classifiers/Results/features_comuni/features_comuni_" + str(SUBJECT) + "_K_" + str(FEATURE_SELECTION_K) + ".csv", mode='w', header=False, index=False)
    
    print("Operazione conclusa in %s seconds \n" % (time.time() - start_time))


if __name__ == "__main__":
    # elenco dei modelli che vogliamo addestrarre
    models = [
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis(solver='svd')),
        # ("Random Forest", RandomForestClassifier(n_estimators=200, criterion='log_loss', max_depth=8, max_features="sqrt", random_state=73, n_jobs=-1)),
        # ("RidgeClassifier", RidgeClassifier(alpha=0.0002, random_state=73)),
        # ("SVM", svm.SVC(C=50, gamma='scale', kernel='linear', random_state=73)),
        # ("MLP", MLPClassifier(solver='adam', activation='identity', hidden_layer_sizes=[30,30,30,30,30, 30,30,30,30,30, 30,30,30], random_state=73)),
        # ("LightGBM", LGBMClassifier(max_depth=15, num_leaves=25, learning_rate=0.05, random_state=73, n_jobs=-1, verbose=-1)),
        ]
    
    # modo banale per far girare tutti i classificatori: non un thread ciascuno ma in successione
    for i in [9]:
        SUBJECT = i
        print("Loading datasets of feature vectors of subject {}...".format(SUBJECT))
        df = pd.read_csv(
             #"D:\Tesi\Datasets\s"
            #+ str(SUBJECT)
             "EMGdataset_"
            + str(SUBJECT)
            + ".feature_vectors_STD.csv"
        )
        print("Dataset loaded. Shape: " + str(df.shape))
        for k in [500]: #15 con 900 e 1000
            FEATURE_SELECTION_K = k

            elements = pre_processing_e_selezione_features(df)

            for m in models:
                classificatore(m[1], elements)
