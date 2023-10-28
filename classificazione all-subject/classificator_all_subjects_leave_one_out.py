#Versione del 2023-08-12 n.1

from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from numpy import random
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler

#librerie dei modelli usati
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier #maximum likelihood estimation
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

#metriche e suddivisione del modello
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

#gestione del sistema
import time

#disablita i warning
import warnings

from xgboost import XGBClassifier, XGBRFClassifier
warnings.filterwarnings("ignore")

# Informazioni per il salvataggio in CSV
SUBJECT = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
SW = 150 # in millisecondi
PREPROCESSING = "STD"
RIDUZIONE = "Kbest"
CANALI_USATI = "nessuno"
OVERLAP = 0
FEATURE_SELECTION_K = 500

def classificatore(df, modello_classificatore: RandomForestClassifier()):
    random.seed(0)

    start_time = time.time()

    f1 = [None] * len(SUBJECT)
    accuracy = [None] * len(SUBJECT)
    precision = [None] * len(SUBJECT)
    recall = [None] * len(SUBJECT)
    confusionMatrix = [None] * len(SUBJECT)
    predict_timer = [None] * len(SUBJECT)
    train_timer = [None] * len(SUBJECT)

    print("\nModello in uso: " + modello_classificatore.__class__.__name__ + "\n")

    for i in range(len(SUBJECT)):

        df_train = pd.DataFrame()
        for j in range(len(SUBJECT)):
            if i != j:
                df_train = pd.concat([df_train, df[j]], ignore_index=True)
        df_test  = df[i]
        print("Dimensioni dataset di training: " + str(df_train.shape) + 
                " | Dimensioni dataset di testing: " + str(df_test.shape))

        #le sottoporzioni del dataset
        X_train = np.array(df_train.iloc[:,:-2]) #rimossa colonna gesto e ripetizione
        y_train = np.array(df_train["class"]) #colonna gesto
        X_test  = np.array(df_test.iloc[:,:-2]) #rimossa colonna gesto e ripetizione
        y_test  = np.array(df_test["class"])  #colonna gesto

        print("soggetto {} ".format(SUBJECT[i])
            + "Dimensioni X_train: " + str(X_train.shape)
            + " | Dimensioni y_train: " + str(y_train.shape)
            + " | Dimensioni X_test: " + str(X_test.shape)
            + " | Dimensioni y_test: " + str(y_test.shape))      
        
        model = modello_classificatore

        #Training
        train_time = time.time()
        model.fit(X_train, y_train)
        train_timer[i] = time.time() - train_time

        # adattamento valori delle classi per XGBClassifier
        if (
            model.__class__.__name__ == "XGBClassifier"
            or model.__class__.__name__ == "XGBRFClassifier"
        ):
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.fit_transform(y_test)

        #Prediction
        predict_time = time.time()
        y_pred = model.predict(X_test)
        predict_timer[i] = time.time() - predict_time

        f1[i] =  f1_score(y_test, y_pred, average="macro") * 100
        accuracy[i] = accuracy_score(y_test, y_pred)  * 100
        precision[i] =  precision_score(y_test, y_pred, average="macro") * 100
        recall[i] = recall_score(y_test, y_pred, average="macro") * 100
        confusionMatrix[i] = confusion_matrix(y_test, y_pred)

        #Results
        print("--------------------------------------------------------\n")
        print("soggetto {} \n".format(SUBJECT[i]) + 
              "Accuracy (fold {}): {:.2f}%\n".format(i, accuracy[i])
            + " | F1 score (fold {}): {:.2f}%\n".format(i, f1[i])
            + " | Precision (fold {}): {:.2f}%\n".format(i, precision[i] )
            + " | Recall (fold {}): {:.2f}%\n".format(i, recall[i]))
        print("--------------------------------------------------------\n\n")
    

    accuracy_avg = np.mean(accuracy)
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1_avg = np.mean(f1)

    print ("Accuracy avg: {:.2f}".format(accuracy_avg))
    print ("Precision avg: {:.2f}".format(precision_avg))
    print ("Recall avg: {:.2f}".format(recall_avg))
    print ("F1 avg: {:.2f}".format(f1_avg))

    # Salvataggio delle informazioni in un csv
    result = pd.DataFrame({'id_soggetti': [SUBJECT],
                            'sw': [SW],
                            'canali_esclusi': [CANALI_USATI],
                            'overlap': [OVERLAP],
                            'op_prepr': [PREPROCESSING],
                            'op_riduz': [RIDUZIONE],
                            'tipologia_cross_validation': ["leave_one_out"],
                            'modello': [modello_classificatore.__class__.__name__],
                            'iperparametri': [modello_classificatore.get_params()],
                            "accuracy": [accuracy],
                            "precision": [precision],
                            "recall": [recall],
                            "f1": [f1],
                            "overall_accuracy": [accuracy_avg],
                            "overall_precision": [precision_avg],
                            "overall_recall": [recall_avg],
                            "overall_f1": [f1_avg],
                            "tempo_addestramento": [train_timer],
                            "tempo_predizione": [predict_timer],
                            "tempo_esecuzione_totale": [float(time.time() - start_time)],
                            }
                        )
    
    result.to_csv('Classifiers/Results/Results_all_subjects_classification.csv', mode='a', header=False, index=False)    

    #salvataggio confusion matrix
    nomi_colonne_cf = [x for x in range(1,66)] #le colonne hanno dei nomi di numeri: 1,2,3,4,5,6,...,65

    # Crea 5 DataFrame dalle matrici
    df_cm_combined = pd.DataFrame()
    for i in range(len(SUBJECT)):
        df_cm = pd.DataFrame(confusionMatrix[0], columns=nomi_colonne_cf)
        df_combined = pd.concat([df_cm_combined, df_cm], ignore_index=True)

    nomeSaveFile = "Classifiers/Results/CM/Results_all_subject_ID_" + str(SUBJECT) + "_leave_one_out_" + model.__class__.__name__ + "_CM.csv" #nome di salvataggio
    df_combined.to_csv(nomeSaveFile, mode='a', header=False, index=False)

    print("Operazione conclusa in %s seconds \n" % (time.time() - start_time))



if __name__ == '__main__':

    #elenco dei modelli che vogliamo addestrarre
    models = [
        #classifictori visti dagli altri paper
        #("Linear Discriminant Analysis", LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.1, n_components=1, tol=0.01)),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis(solver='svd')),
        #("Random Forest", RandomForestClassifier(n_estimators=200, criterion='log_loss', max_depth=8, max_features="sqrt", random_state=73, n_jobs=-1)),
        #("RidgeClassifier", RidgeClassifier(alpha=0.0002, random_state=73)),
        
        #("SVM", svm.SVC(C=50, gamma='scale', kernel='linear', random_state=73)),
        #("MLP", MLPClassifier(solver='adam', activation='identity', hidden_layer_sizes=[30,30,30,30,30, 30,30,30,30,30, 30,30,30], random_state=73)),
        #("LightGBM", LGBMClassifier(max_depth=15, num_leaves=25, learning_rate=0.05, random_state=73, n_jobs=-1, verbose=-1)),
    ]

    #caricamento dataset originale
    print("Loading datasets of feature vectors...")
    df_temp = [None] * len(SUBJECT)
    for i in range(len(SUBJECT)):
        nomefile = "EMGdataset_" + str(SUBJECT[i]) +  ".feature_vectors_STD_K_500.csv"
        df_temp[i] = pd.read_csv(nomefile)
        print("Standardizzazione dei dati...")
        X = df_temp[i].iloc[:,:-2]
        y = df_temp[i]["class"]
        id = df_temp[i]["id"]
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        df_temp[i] = pd.DataFrame(X)
        df_temp[i]["class"] = y
        df_temp[i]["id"] = id
        print("Standardizzazione completata")
        print("Dataset of subject {} loaded with shape: {}".format(SUBJECT[i], df_temp[i].shape))

    # colonne = df_temp[0].columns

    # df_completo = pd.DataFrame()
    # for i in range(len(SUBJECT)):
    #     df_completo = pd.concat([df_completo, df_temp[i]], ignore_index=True)
    # print("Dataset completo creato con shape: ", df_completo.shape)

    # print("Standardizzazione dei dati...")
    # X = df_completo.iloc[:,:-2]
    # y = df_completo["class"]
    # id = df_completo["id"]
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    # df_completo_std = pd.DataFrame(X)
    # df_completo_std["class"] = y
    # df_completo_std["id"] = id
    # print("Standardizzazione completata")

    # print("Divisione del dataset in {} parti...".format(len(SUBJECT)))
    # for i in range(len(SUBJECT) - 1):
    #     if(i == 0):
    #         df_temp[i] = df_completo_std.iloc[0:df_temp[i].shape[0]]
    #         df_temp[i].columns = colonne

    #     else:
    #         df_temp[i] = df_completo_std.iloc[sum([df_temp[j].shape[0] for j in range(i)]):sum([df_temp[j].shape[0] for j in range(i+1)])]
    #         df_temp[i].columns = colonne

    #modo banale per far girare tutti i classificatori: non un thread ciascuno ma in successione
    for m in models:
        classificatore(df_temp, m[1])

    

