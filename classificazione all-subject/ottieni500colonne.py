
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest

FEATURE_SELECTION_K = 5000
elenco = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
path = "feature_comuni_tutti_soggetti_3.csv"

#costruisco un insieme con gli indici delle feature comuni selezionate nelle 5 fold
lista_nomi_features = pd.read_csv('new_labels.csv', header=None)

df_completo = pd.DataFrame()
for i in range(len(elenco)):
    print(elenco[i])
    #path = "Classifiers/Results/features_comuni/features_comuni_" + str(i) + "_K_1000.csv"
    #cols_idxs[i]  = pd.read_csv(path).columns.values.tolist()
    #faccio l'intersezioni per ottenerle le feature comuni a tutte le fold
    #cols_idxs[i] = cols_idxs[i][0]
    df = pd.read_csv("EMGdataset_" + str(elenco[i]) + ".feature_vectors_STD.csv")
    df_completo = pd.concat([df_completo, df], ignore_index=True)
    print(df_completo.shape)

feature_comuni_fold = [None] * 5
for j in range(1, 6):
    df_train = df[df["id"] != j]
    df_test = df[df["id"] == j]

    # le sottoporzioni del dataset
    X_train = np.array(df_train.iloc[:, :-2])  # rimossa colonna gesto e ripetizione
    y_train = np.array(df_train["class"])  # colonna gesto
    X_test = np.array(df_test.iloc[:, :-2])  # rimossa colonna gesto e ripetizione
    y_test = np.array(df_test["class"])  # colonna gesto

    #selezione delle k feature migliori
    selector = SelectKBest(k=FEATURE_SELECTION_K)
    X_train = selector.fit_transform(X_train, y_train)
    # Get columns to keep and create new dataframe with those only
    feature_comuni_fold[j-1] = selector.get_support(indices=True)
    X_test = X_test[:,feature_comuni_fold[j-1]]

#print("Selezione completata in {:.2f} secondi".format(sum(selection_time_stamps)))
#print(feature_comuni_fold)
#costruisco un insieme con gli indici delle feature comuni selezionate nelle 5 fold

feature_id_comuni = feature_comuni_fold[0]
#faccio l'intersezioni per ottenerle le feature comuni a tutte le fold
for k in range(1,5):
    feature_id_comuni = list(set(feature_id_comuni) & set(feature_comuni_fold[k]))

df_feture_comuni = pd.DataFrame(feature_id_comuni).transpose()
df_feture_comuni.to_csv(path, header=None)

print("Numero di feature comuni: {}".format(len(feature_id_comuni))) 

new_labels = pd.read_csv("new_labels.csv", header=None)
feature_comuni_tutti_soggetti_id = pd.read_csv(path, header=None).to_numpy()[0]
print(feature_comuni_tutti_soggetti_id)
labels_comuni_tutti_soggetti = new_labels.iloc[:, feature_comuni_tutti_soggetti_id]

labels_comuni_tutti_soggetti.to_csv(path, index=False)