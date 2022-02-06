# python3.9 main.py

import pandas
import os
import sklearn
import numpy
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from numpy import set_printoptions

import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from tabulate import tabulate

# Stage 1 - SelectKBest, colours and C-style string experiments
################################################################################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

COLUMNS_NAMES = ("Temperatura",
                 "Anemia",
                 "Stopień krwawienia",
                 "Miejsce krwawienia",
                 "Bóle kości",
                 "Wrażliwość mostka",
                 "Powiększenie węzłów chłonnych",
                 "Powiększenie wątroby i śledziony",
                 "Centralny układ nerwowy", #(ból głowy, wymioty, drgawki, senność, śpiączka)",
                 "Powiększenie jąder",
                 "Uszkodzenie w sercu, płucach, nerce",
                 "Gałka oczna",  #(zaburzenia w widzeniu, krwawienie do siatkówki, wytrzeszcz oczu)",
                 "Poziom WBC (leukocytów)",
                 "Obniżenie licby RRC (erytrocytów)",
                 "Liczba płytek krwi",
                 "Niedojrzałe komórki (blastyczne)",
                 "Stan pobudzenia szpiku",
                 "Główne komórki w szpiku",
                 "Poziom limfocytów",
                 "Reakcja (chemiczna tkanki)",
                 "Diagnoza")

os.system("clear")

print(f"{bcolors.OKBLUE}Komputerowe wspomaganie diagnozowania białaczek u dzieci z wykorzystaniem algorytmu{bcolors.ENDC}", end = '')
print(f"{bcolors.FAIL} K_NN{bcolors.ENDC}")
print(f"{bcolors.HEADER}Autor: {bcolors.ENDC}", end = '')
print(f"{bcolors.BOLD}Tobiasz Puślecki{bcolors.ENDC}", end = '')
print(f"{bcolors.WARNING} 241354{bcolors.ENDC}")

data = pandas.read_csv("leukemia_data.csv", names=COLUMNS_NAMES, delimiter=";")
X = data.values[1:,:-1] # take all data except the last col (diagnosis) and first row
y = data.values[1:,-1]  # all except last col (diagnosis)

fit = SelectKBest(score_func=chi2).fit(X, y)

print(f"{bcolors.OKCYAN}\n\nWyniki funkcji SelectKBest(chi2):\n{bcolors.ENDC}")

pairs = [()]

for i in range(1,len(fit.scores_)+1):
    fixed_string_1 = "{0:>4}".format(str(i))
    fixed_string_2 = "{0:<35}".format(str(COLUMNS_NAMES[i]))
    print(fixed_string_1 + ". " + fixed_string_2 + " | " + str(round(fit.scores_[i-1],3)))

    pairs.append((i,fit.scores_[i-1]))

pairs.pop(0)

# Function to sort by second item
sorted = sorted(pairs, key = lambda x: x[1], reverse=True)

# select N best
number_of_features = 10
print(f"{bcolors.OKCYAN}\n\nPosortowane N=10 najlepszych cech:\n{bcolors.ENDC}")

for i in range(0, number_of_features):
    fixed_string_1 = "{0:>4}".format(str(sorted[i][0]))
    fixed_string_2 = "{0:<35}".format(str(COLUMNS_NAMES[sorted[i][0]]))
    print(fixed_string_1 + ". " + fixed_string_2 + " | " + str(round(sorted[i][1],3)))

print("\n\n")

# Stage 2 - Experiment and statistical stuff
################################################################################

clfs = {
    'kNN(1-e)': KNeighborsClassifier(n_neighbors=1, metric='euclidean'),
    'kNN(5-e)': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    'kNN(10-e)': KNeighborsClassifier(n_neighbors=10, metric='euclidean'),
    'kNN(1-m)': KNeighborsClassifier(n_neighbors=1, metric='manhattan'),
    'kNN(5-m)': KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
    'kNN(10-m)': KNeighborsClassifier(n_neighbors=10, metric='manhattan'),
}

leukemia_classes = data.iloc[:, 20:21].values

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

n_splits = 2
n_repeats = 5
scores = np.zeros((len(clfs), number_of_features, n_splits * n_repeats))

for features_index in range(0,number_of_features):
  selected_data = SelectKBest(score_func=chi2, k=features_index + 1).fit_transform(X, y)
  for fold_id, (train_index, test_index) in enumerate(rskf.split(selected_data, y)):
      for clf_id, clf_name in enumerate(clfs):
          X_train, X_test = selected_data[train_index], selected_data[test_index]
          y_train, y_test = leukemia_classes[train_index], leukemia_classes[test_index]
          # To silent DataConversionWarning
          y_train_ravel = np.ravel(y_train)
          clf = clone(clfs[clf_name])
          clf.fit(X_train, y_train_ravel)
          y_pred = clf.predict(X_test)
          scores[clf_id, features_index, fold_id] = accuracy_score(y_test, y_pred)

means = np.mean(scores,axis=2)
stds = np.std(scores,axis=2)

print(f"{bcolors.OKCYAN}Klasyfikator , ilość cech, średnia, odchylenie std.{bcolors.ENDC}\n")
for clf_id, clf_name in enumerate(clfs):
  print(f"{bcolors.HEADER}klasyfikator:{clf_name}{bcolors.ENDC}")
  for feature_index in range(0,number_of_features):
    current_classifier_mean = means[clf_id,feature_index]
    print("ilość cech: %d, średnia: %.3f, odchylenie: %.2f" % (feature_index+1, current_classifier_mean, stds[clf_id,feature_index]))

best_accurancy, best_classifier_id,best_feature_index = np.max(means), np.argmax(np.max(means, axis=1)), np.argmax(np.max(means, axis=0))
print(f"\n\n{bcolors.FAIL}Najlepszy: {list(clfs.keys())[best_classifier_id]}, wynik {round(best_accurancy,3)}, ilość cech {best_feature_index + 1}{bcolors.ENDC}\n\n")

t_statistic_matrixes = []
p_statistic_matrixes = []

for _ in range(number_of_features):
  t_statistic_matrixes.append(np.zeros((len(clfs), len(clfs))))
  p_statistic_matrixes.append(np.zeros((len(clfs), len(clfs))))

best_N_features_results = []

for feature_index in range(number_of_features):
  for clf_id in range(len(clfs)):
    best_N_features_results.append(scores[clf_id,feature_index])

for matrix_id in range(number_of_features):
  for i in range(len(clfs)):
    for j in range(len(clfs)):
      t_statistic_matrixes[matrix_id][i][j], p_statistic_matrixes[matrix_id][i][j] = ttest_ind(best_N_features_results[i + matrix_id * len(clfs)], best_N_features_results[j + matrix_id * len(clfs)])
      # print(best_N_features_results[j + matrix_id * len(clfs)])


headers = ["kNN(1-e)", "kNN(5-e)", "kNN(10-e)", "kNN(1-m)", "kNN(5-m)", "kNN(10-m)"]
names_column = np.array([["kNN(1-e)"], ["kNN(5-e)"], ["kNN(10-e)"], ["kNN(1-m)"], ["kNN(5-m)"], ["kNN(10-m)"]])
for i in range(number_of_features):
    current_t_matrix = t_statistic_matrixes[i]
    t_statistic_table = np.concatenate((names_column, current_t_matrix), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    print(f"\n{bcolors.OKCYAN}t-statistic for {i+1} feature:{bcolors.ENDC}\n", t_statistic_table)

advantages = []

for _ in range(number_of_features):
  advantages.append(np.zeros((len(clfs), len(clfs))))

for index,t_statistic in enumerate(t_statistic_matrixes):
  advantages[index][t_statistic > 0] = 1

for i in range(number_of_features):
    current_adv_matrix = advantages[i]
    advantages_table = np.concatenate((names_column, current_adv_matrix), axis=1)
    advantages_table = tabulate(advantages_table, headers)
    print(f"{bcolors.OKCYAN}\nPrzewaga dla {i+1} cech:{bcolors.ENDC}\n", advantages_table)

alfa = .05
significances = []

for _ in range(number_of_features):
  significances.append(np.zeros((len(clfs), len(clfs))))

for index,p_value in enumerate(p_statistic_matrixes):
  significances[index][p_value <= alfa] = 1

for i in range(number_of_features):
    current_p_matrix = p_statistic_matrixes[i]
    p_value_table = np.concatenate((names_column, current_p_matrix), axis=1)
    p_value_table = tabulate(p_value_table, headers)
    print(f"{bcolors.OKCYAN}\np-value dla {i+1} cech:{bcolors.ENDC}\n", p_value_table)

for i in range(number_of_features):
    current_sig_matrix = significances[i]
    sig_table = np.concatenate((names_column, current_sig_matrix), axis=1)
    sig_table = tabulate(sig_table, headers)
    print(f"{bcolors.OKCYAN}\nRóżnice statystycznie znaczące (alpha = 0.05) dla {i+1} cech:{bcolors.ENDC}\n", sig_table)

stat_better_matrixes = []

for i in range(number_of_features):
  stat_better = significances[i] * advantages[i]
  stat_better_matrixes.append(stat_better)

for i in range(number_of_features):
    current_sig_bet_matrix = stat_better_matrixes[i]
    sig_bet_table = np.concatenate((names_column, current_sig_bet_matrix), axis=1)
    sig_bet_table = tabulate(sig_bet_table, headers)
    print(f"{bcolors.OKCYAN}\nStatystycznie znacząco lepszy dla {i+1} cech:{bcolors.ENDC}\n", sig_bet_table)




#
