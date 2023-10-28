import pandas as pd
import numpy as np

#load top results.csv

df = pd.read_csv('top_results.csv',index_col=None)

#print(df.head())

# ("LogisticRegression:         ", np.mean(df[df['modello'] == "LogisticRegression"]['overall_f1'])),
#     ("RidgeClassifier:            ", np.mean(df[df['modello'] == "RidgeClassifier"]['overall_f1'])),
#     ("SVC:                        ", np.mean(df[df['modello'] == "SVC"]['overall_f1'])),
#     ("XGBClassifier:              ", np.mean(df[df['modello'] == "XGBClassifier"]['overall_f1'])),
#     ("RandomForestClassifier:     ", np.mean(df[df['modello'] == "RandomForestClassifier"]['overall_f1'])),
#     ("LGBMClassifier:             ", np.mean(df[df['modello'] == "LGBMClassifier"]['overall_f1'])),
#     ("MLPClassifier:              ", np.mean(df[df['modello'] == "MLPClassifier"]['overall_f1'])),
#     ("KNeighborsClassifier:       ", np.mean(df[df['modello'] == "KNeighborsClassifier"]['overall_f1'])),
#     ("XGBRFClassifier:            ", np.mean(df[df['modello'] == "XGBRFClassifier"]['overall_f1'])),
#     ("DecisionTreeClassifier:     ", np.mean(df[df['modello'] == "DecisionTreeClassifier"]['overall_f1']))

classificatori = [
    ("LinearDiscriminantAnalysis: ", np.mean(df[df['modello'] == "LinearDiscriminantAnalysis"]['overall_f1']), np.max(df[df['modello'] == "LinearDiscriminantAnalysis"]['overall_f1']), np.min(df[df['modello'] == "LinearDiscriminantAnalysis"]['overall_f1'])),
    ##("LogisticRegression:         ", np.mean(df[df['modello'] == "LogisticRegression"]['overall_f1']), np.max(df[df['modello'] == "LogisticRegression"]['overall_f1']), np.min(df[df['modello'] == "LogisticRegression"]['overall_f1'])),
    ("RidgeClassifier:            ", np.mean(df[df['modello'] == "RidgeClassifier"]['overall_f1']), np.max(df[df['modello'] == "RidgeClassifier"]['overall_f1']), np.min(df[df['modello'] == "RidgeClassifier"]['overall_f1'])),
    ("SVC:                        ", np.mean(df[df['modello'] == "SVC"]['overall_f1']), np.max(df[df['modello'] == "SVC"]['overall_f1']), np.min(df[df['modello'] == "SVC"]['overall_f1'])),
    ##("XGBClassifier:              ", np.mean(df[df['modello'] == "XGBClassifier"]['overall_f1']), np.max(df[df['modello'] == "XGBClassifier"]['overall_f1']), np.min(df[df['modello'] == "XGBClassifier"]['overall_f1'])),
    ("RandomForestClassifier:     ", np.mean(df[df['modello'] == "RandomForestClassifier"]['overall_f1']), np.max(df[df['modello'] == "RandomForestClassifier"]['overall_f1']), np.min(df[df['modello'] == "RandomForestClassifier"]['overall_f1'])),
    ("LGBMClassifier:             ", np.mean(df[df['modello'] == "LGBMClassifier"]['overall_f1']), np.max(df[df['modello'] == "LGBMClassifier"]['overall_f1']), np.min(df[df['modello'] == "LGBMClassifier"]['overall_f1'])),
    ("MLPClassifier:              ", np.mean(df[df['modello'] == "MLPClassifier"]['overall_f1']), np.max(df[df['modello'] == "MLPClassifier"]['overall_f1']), np.min(df[df['modello'] == "MLPClassifier"]['overall_f1'])),
    ##("KNeighborsClassifier:       ", np.mean(df[df['modello'] == "KNeighborsClassifier"]['overall_f1']), np.max(df[df['modello'] == "KNeighborsClassifier"]['overall_f1']), np.min(df[df['modello'] == "KNeighborsClassifier"]['overall_f1'])),
    #("XGBRFClassifier:            ", np.mean(df[df['modello'] == "XGBRFClassifier"]['overall_f1']), np.max(df[df['modello'] == "XGBRFClassifier"]['overall_f1']), np.min(df[df['modello'] == "XGBRFClassifier"]['overall_f1'])),
    ##("DecisionTreeClassifier:     ", np.mean(df[df['modello'] == "DecisionTreeClassifier"]['overall_f1']), np.max(df[df['modello'] == "DecisionTreeClassifier"]['overall_f1']), np.min(df[df['modello'] == "DecisionTreeClassifier"]['overall_f1']))
    
]
print("Classificatori ordinati per media f1, max f1 e min f1")
#print dei classificatori riordinati per il secondo parametro (media f1)
classificatori.sort(key=lambda x: x[1], reverse=True)
for i in classificatori:
    print(i)



#classificzione per id soggetto
soggetti = [
    ("Soggetto n.1:     ", np.mean(df[df['id_soggetto'] == 1]['overall_f1']), np.max(df[df['id_soggetto'] == 1]['overall_f1']), np.min(df[df['id_soggetto'] == 1]['overall_f1'])),
    ("Soggetto n.2:     ", np.mean(df[df['id_soggetto'] == 2]['overall_f1']), np.max(df[df['id_soggetto'] == 2]['overall_f1']), np.min(df[df['id_soggetto'] == 2]['overall_f1'])),
    ("Soggetto n.3:     ", np.mean(df[df['id_soggetto'] == 3]['overall_f1']), np.max(df[df['id_soggetto'] == 3]['overall_f1']), np.min(df[df['id_soggetto'] == 3]['overall_f1'])),
    ("Soggetto n.4:     ", np.mean(df[df['id_soggetto'] == 4]['overall_f1']), np.max(df[df['id_soggetto'] == 4]['overall_f1']), np.min(df[df['id_soggetto'] == 4]['overall_f1'])),
    ("Soggetto n.6:     ", np.mean(df[df['id_soggetto'] == 6]['overall_f1']), np.max(df[df['id_soggetto'] == 6]['overall_f1']), np.min(df[df['id_soggetto'] == 6]['overall_f1'])),
    ("Soggetto n.7:     ", np.mean(df[df['id_soggetto'] == 7]['overall_f1']), np.max(df[df['id_soggetto'] == 7]['overall_f1']), np.min(df[df['id_soggetto'] == 7]['overall_f1'])),
    ("Soggetto n.8:     ", np.mean(df[df['id_soggetto'] == 8]['overall_f1']), np.max(df[df['id_soggetto'] == 8]['overall_f1']), np.min(df[df['id_soggetto'] == 8]['overall_f1'])),
    ("Soggetto n.9:     ", np.mean(df[df['id_soggetto'] == 9]['overall_f1']), np.max(df[df['id_soggetto'] == 9]['overall_f1']), np.min(df[df['id_soggetto'] == 9]['overall_f1'])),
    ("Soggetto n.10:    ", np.mean(df[df['id_soggetto'] == 10]['overall_f1']), np.max(df[df['id_soggetto'] == 10]['overall_f1']), np.min(df[df['id_soggetto'] == 10]['overall_f1'])),
    ("Soggetto n.11:    ", np.mean(df[df['id_soggetto'] == 11]['overall_f1']), np.max(df[df['id_soggetto'] == 11]['overall_f1']), np.min(df[df['id_soggetto'] == 11]['overall_f1'])),
    ("Soggetto n.12:    ", np.mean(df[df['id_soggetto'] == 12]['overall_f1']), np.max(df[df['id_soggetto'] == 12]['overall_f1']), np.min(df[df['id_soggetto'] == 12]['overall_f1'])),
    ("Soggetto n.13:    ", np.mean(df[df['id_soggetto'] == 13]['overall_f1']), np.max(df[df['id_soggetto'] == 13]['overall_f1']), np.min(df[df['id_soggetto'] == 13]['overall_f1'])),
    ("Soggetto n.14:    ", np.mean(df[df['id_soggetto'] == 14]['overall_f1']), np.max(df[df['id_soggetto'] == 14]['overall_f1']), np.min(df[df['id_soggetto'] == 14]['overall_f1'])),
    ("Soggetto n.15:    ", np.mean(df[df['id_soggetto'] == 15]['overall_f1']), np.max(df[df['id_soggetto'] == 15]['overall_f1']), np.min(df[df['id_soggetto'] == 15]['overall_f1'])),
    ("Soggetto n.16:    ", np.mean(df[df['id_soggetto'] == 16]['overall_f1']), np.max(df[df['id_soggetto'] == 16]['overall_f1']), np.min(df[df['id_soggetto'] == 16]['overall_f1'])),
    ("Soggetto n.17:    ", np.mean(df[df['id_soggetto'] == 17]['overall_f1']), np.max(df[df['id_soggetto'] == 17]['overall_f1']), np.min(df[df['id_soggetto'] == 17]['overall_f1'])),
    ("Soggetto n.18:    ", np.mean(df[df['id_soggetto'] == 18]['overall_f1']), np.max(df[df['id_soggetto'] == 18]['overall_f1']), np.min(df[df['id_soggetto'] == 18]['overall_f1'])),
    ("Soggetto n.19:    ", np.mean(df[df['id_soggetto'] == 19]['overall_f1']), np.max(df[df['id_soggetto'] == 19]['overall_f1']), np.min(df[df['id_soggetto'] == 19]['overall_f1'])),
    ("Soggetto n.20:    ", np.mean(df[df['id_soggetto'] == 20]['overall_f1']), np.max(df[df['id_soggetto'] == 20]['overall_f1']), np.min(df[df['id_soggetto'] == 20]['overall_f1']))
]

print("Soggetti ordinati per media f1, max f1 e min f1")
soggetti.sort(key=lambda x: x[1], reverse=True)
for i in soggetti:
    print(i)
#media delle medie dei soggetti
#print("Media delle medie dei soggetti: ", np.mean([i[1] for i in soggetti]))


#classificazion per valore di k

k = [
    ("k = 100:     ", np.mean(df[df['feature_selection'] == 100]['overall_f1']), np.max(df[df['feature_selection'] == 100]['overall_f1']), np.min(df[df['feature_selection'] == 100]['overall_f1'])),
    ("k = 200:     ", np.mean(df[df['feature_selection'] == 200]['overall_f1']), np.max(df[df['feature_selection'] == 200]['overall_f1']), np.min(df[df['feature_selection'] == 200]['overall_f1'])),
    ("k = 300:     ", np.mean(df[df['feature_selection'] == 300]['overall_f1']), np.max(df[df['feature_selection'] == 300]['overall_f1']), np.min(df[df['feature_selection'] == 300]['overall_f1'])),
    ("k = 400:     ", np.mean(df[df['feature_selection'] == 400]['overall_f1']), np.max(df[df['feature_selection'] == 400]['overall_f1']), np.min(df[df['feature_selection'] == 400]['overall_f1'])),
    ("k = 500:     ", np.mean(df[df['feature_selection'] == 500]['overall_f1']), np.max(df[df['feature_selection'] == 500]['overall_f1']), np.min(df[df['feature_selection'] == 500]['overall_f1'])),
    ("k = 600:     ", np.mean(df[df['feature_selection'] == 600]['overall_f1']), np.max(df[df['feature_selection'] == 600]['overall_f1']), np.min(df[df['feature_selection'] == 600]['overall_f1'])),
    ("k = 700:     ", np.mean(df[df['feature_selection'] == 700]['overall_f1']), np.max(df[df['feature_selection'] == 700]['overall_f1']), np.min(df[df['feature_selection'] == 700]['overall_f1'])),
    ("k = 800:     ", np.mean(df[df['feature_selection'] == 800]['overall_f1']), np.max(df[df['feature_selection'] == 800]['overall_f1']), np.min(df[df['feature_selection'] == 800]['overall_f1'])),
    ("k = 900:    ",  np.mean(df[df['feature_selection'] == 900]['overall_f1']), np.max(df[df['feature_selection'] == 900]['overall_f1']), np.min(df[df['feature_selection'] == 900]['overall_f1'])),
    ("k =1000:    ",  np.mean(df[df['feature_selection'] == 1000]['overall_f1']), np.max(df[df['feature_selection'] == 1000]['overall_f1']), np.min(df[df['feature_selection'] == 1000]['overall_f1']))
]

print("Soggetti ordinati per media f1, max f1 e min f1")
k.sort(key=lambda x: x[1], reverse=True)
for i in k:
    print(i)


#per ordine di tempo di esecuzione

classificatori = [
    ("LinearDiscriminantAnalysis: ", np.mean(df[df['modello'] == "LinearDiscriminantAnalysis"]['tempo_esecuzione_totale']), np.max(df[df['modello'] == "LinearDiscriminantAnalysis"]['tempo_esecuzione_totale']), np.min(df[df['modello'] == "LinearDiscriminantAnalysis"]['tempo_esecuzione_totale'])),
    ##("LogisticRegression:         ", np.mean(df[df['modello'] == "LogisticRegression"]['overall_f1']), np.max(df[df['modello'] == "LogisticRegression"]['overall_f1']), np.min(df[df['modello'] == "LogisticRegression"]['overall_f1'])),
    ("RidgeClassifier:            ", np.mean(df[df['modello'] == "RidgeClassifier"]['tempo_esecuzione_totale']), np.max(df[df['modello'] == "RidgeClassifier"]['tempo_esecuzione_totale']), np.min(df[df['modello'] == "RidgeClassifier"]['tempo_esecuzione_totale'])),
    ("SVC:                        ", np.mean(df[df['modello'] == "SVC"]['tempo_esecuzione_totale']), np.max(df[df['modello'] == "SVC"]['tempo_esecuzione_totale']), np.min(df[df['modello'] == "SVC"]['tempo_esecuzione_totale'])),
    ##("XGBClassifier:              ", np.mean(df[df['modello'] == "XGBClassifier"]['overall_f1']), np.max(df[df['modello'] == "XGBClassifier"]['overall_f1']), np.min(df[df['modello'] == "XGBClassifier"]['overall_f1'])),
    ("RandomForestClassifier:     ", np.mean(df[df['modello'] == "RandomForestClassifier"]['tempo_esecuzione_totale']), np.max(df[df['modello'] == "RandomForestClassifier"]['tempo_esecuzione_totale']), np.min(df[df['modello'] == "RandomForestClassifier"]['tempo_esecuzione_totale'])),
    ("LGBMClassifier:             ", np.mean(df[df['modello'] == "LGBMClassifier"]['tempo_esecuzione_totale']), np.max(df[df['modello'] == "LGBMClassifier"]['tempo_esecuzione_totale']), np.min(df[df['modello'] == "LGBMClassifier"]['tempo_esecuzione_totale'])),
    ("MLPClassifier:              ", np.mean(df[df['modello'] == "MLPClassifier"]['tempo_esecuzione_totale']), np.max(df[df['modello'] == "MLPClassifier"]['tempo_esecuzione_totale']), np.min(df[df['modello'] == "MLPClassifier"]['tempo_esecuzione_totale'])),
    ##("KNeighborsClassifier:       ", np.mean(df[df['modello'] == "KNeighborsClassifier"]['overall_f1']), np.max(df[df['modello'] == "KNeighborsClassifier"]['overall_f1']), np.min(df[df['modello'] == "KNeighborsClassifier"]['overall_f1'])),
    #("XGBRFClassifier:            ", np.mean(df[df['modello'] == "XGBRFClassifier"]['overall_f1']), np.max(df[df['modello'] == "XGBRFClassifier"]['overall_f1']), np.min(df[df['modello'] == "XGBRFClassifier"]['overall_f1'])),
    ##("DecisionTreeClassifier:     ", np.mean(df[df['modello'] == "DecisionTreeClassifier"]['overall_f1']), np.max(df[df['modello'] == "DecisionTreeClassifier"]['overall_f1']), np.min(df[df['modello'] == "DecisionTreeClassifier"]['overall_f1']))
    
]
print("Classificatori ordinati per media tempo, max tempo e min tempo")
#print dei classificatori riordinati per il secondo parametro (media f1)
classificatori.sort(key=lambda x: x[1], reverse=True)
for i in classificatori:
    print(i)


