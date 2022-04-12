import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import math
from collections import Counter
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import plot_confusion_matrix, confusion_matrix


sns.set()


# Finds nearest neighbour. Calculates hypotenuse for all data points and adds new column
def neighbour(my_length, my_weight, data):
    nearest = []

    for length, weight in zip(data['stature'], data['weightkg']):
        delta_y = abs(my_length) - abs(length)
        delta_x = abs(my_weight) - abs(weight)
        hypo = delta_y ** 2 + delta_x ** 2

        nearest.append(math.sqrt(hypo))

    data['K'] = nearest


# Adds t-shirt size to male dataset
def add_t_size_male(data):
    size_t = []

    for chest, waist in zip(data['chestcircumference'], data['waistcircumference']):
        if chest < 880:
            if waist >= 760:
                size_t.append('Small')
            else:
                size_t.append('X-Small')
        elif chest < 960:
            if waist >= 840:
                size_t.append('Medium')
            else:
                size_t.append('Small')
        elif chest < 1040:
            if waist >= 920:
                size_t.append('Large')
            else:
                size_t.append('Medium')
        elif chest < 1120:
            if waist >= 1000:
                size_t.append('X-Large')
            else:
                size_t.append('Large')
        elif chest < 1200:
            if waist >= 1080:
                size_t.append('XX-Large')
            else:
                size_t.append('X-Large')
        elif chest < 1270:
            size_t.append('XX-Large')
        elif chest < 1345:
            size_t.append('XXX-Large')
        elif chest < 1500:
            size_t.append('XXXX-Large')

    data['Size_t'] = size_t


# Adds t-shirt size to female dataset
def add_t_size_female(data):
    size_t = []

    for chest, waist in zip(data['chestcircumference'], data['waistcircumference']):
        if chest < 800:
            if waist >= 640:
                size_t.append('X-Small')
            else:
                size_t.append('XX-Small')
        elif chest < 840:
            if waist >= 680:
                size_t.append('Small')
            else:
                size_t.append('X-Small')
        elif chest < 920:
            if waist >= 760:
                size_t.append('Medium')
            else:
                size_t.append('Small')
        elif chest < 1000:
            if waist >= 850:
                size_t.append('Large')
            else:
                size_t.append('Medium')
        elif chest < 1100:
            if waist >= 960:
                size_t.append('X-Large')
            else:
                size_t.append('Large')
        elif chest < 1220:
            if waist >= 1080:
                size_t.append('XX-Large')
            else:
                size_t.append('X-Large')
        elif chest < 1340:
            size_t.append('XXX-Large')

    data['Size_t'] = size_t


# Adds pant size to male dataset
def add_p_size_male(data):
    size_p = []

    for waist, butt, crotch in zip(data['waistcircumference'], data['buttockcircumference'], data['crotchheight']):
        if waist < 760:
            if butt >= 930 or crotch >= 810:
                size_p.append('Small')
            else:
                size_p.append('X-Small')
        elif waist < 840:
            if butt >= 990 or crotch >= 820:
                size_p.append('Medium')
            else:
                size_p.append('Small')
        elif waist < 920:
            if butt >= 1050 or crotch >= 830:
                size_p.append('Large')
            else:
                size_p.append('Medium')
        elif waist < 1000:
            if butt >= 1100 or crotch >= 850:
                size_p.append('X-Large')
            else:
                size_p.append('Large')
        elif waist < 1080:
            if butt >= 1170 or crotch >= 860:
                size_p.append('XX-Large')
            else:
                size_p.append('X-Large')
        elif 1080 <= waist < 1120:
            size_p.append('XX-Large')
        elif waist >= 1120:
            size_p.append('XXX-Large')

    data['Size_p'] = size_p


# Adds pant size to female dataset
def add_p_size_female(data):
    size_p = []

    for waist, butt in zip(data['waistcircumference'], data['buttockcircumference']):
        if waist < 640:
            if butt >= 880:
                size_p.append('X-Small')
            else:
                size_p.append('XX-Small')
        elif waist < 680:
            if butt >= 920:
                size_p.append('Small')
            else:
                size_p.append('X-Small')
        elif waist < 760:
            if butt >= 990:
                size_p.append('Medium')
            else:
                size_p.append('Small')
        elif waist < 850:
            if butt >= 1050:
                size_p.append('Large')
            else:
                size_p.append('Medium')
        elif waist < 960:
            if butt >= 1130:
                size_p.append('X-Large')
            else:
                size_p.append('Large')
        elif waist < 1080:
            if butt >= 1230:
                size_p.append('XX-Large')
            else:
                size_p.append('X-Large')
        elif waist < 1210:
            if butt >= 1340:
                size_p.append('XXX-Large')
            else:
                size_p.append('XX-Large')
        elif waist < 1350:
            size_p.append('XXX-Large')

    data['Size_p'] = size_p


# Plots both t-shirt and pants graph. Divides stature and weight by 10 for better readability
def plot_graph(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'XX-Small': 'purple', 'X-Small': 'orange', 'Small': 'red', 'Medium': 'blue', 'Large': 'yellow',
              'X-Large': 'green', 'XX-Large': 'black', 'XXX-Large': 'pink', 'XXXX-Large': 'brown'}
    plt.subplot(1, 2, 1)
    sns.scatterplot(y=data['stature'] / 10, x=data['weightkg'] / 10, c=data['Size_t'].map(colors));
    plt.xlabel('Weight in kg')
    plt.ylabel('Height in cm')
    plt.title("T-Shirt sizes")

    plt.subplot(1, 2, 2)
    sns.scatterplot(y=data['stature'] / 10, x=data['weightkg'] / 10, c=data['Size_p'].map(colors));
    plt.xlabel('Weight in kg')
    plt.ylabel('Height in cm')
    plt.title("Pant sizes")
    plt.show()


# Gets most frequent size from dataset
def most_frequent(size_list):
    occurrence_count = Counter(size_list)
    return occurrence_count.most_common(1)[0][0]


# Uses sklearn predict for sizes. Converts x and y inputs to numpy array.
def kneigbor_predict(data, length, weight):
    combined_list = list(zip(data['stature'], data['weightkg']))

    X = np.array(combined_list)
    y_t = np.array(data['Size_t'])
    y_p = np.array(data['Size_p'])

    X.reshape(1, -1)

    knn_t = KNeighborsClassifier(n_neighbors=5)
    knn_p = KNeighborsClassifier(n_neighbors=5)
    knn_t.fit(X, y_t)
    knn_p.fit(X, y_p)

    predicted_t = ' '.join(map(str, knn_t.predict([[length, weight]])))
    predicted_p = ' '.join(map(str, knn_p.predict([[length, weight]])))

    print(f'\nKNeighborsClassifier predicted you should wear t-shirt size: {predicted_t}')
    print(f'KNeighborsClassifier predicted you should wear pant size: {predicted_p}')


def dtree_predict(data, length, weight):
    combined_list = list(zip(data['stature'], data['weightkg']))

    X = np.array(combined_list)
    # tshirt
    y_t = np.array(data['Size_t'])
    # pants
    y_p = np.array(data['Size_p'])

    X.reshape(1, -1)

    # two training sets for tshirt and pants
    X_train, X_test, y_train, y_test = train_test_split(X, y_t, random_state=42, test_size=0.33)
    X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X, y_p, random_state=42, test_size=0.33)

    clf_dt_pruned_t = DecisionTreeClassifier(random_state=42, ccp_alpha=0.0013)
    clf_dt_pruned_t = clf_dt_pruned_t.fit(X_train, y_train)
    clf_dt_pruned_p = DecisionTreeClassifier(random_state=42, ccp_alpha=0.0009)
    clf_dt_pruned_p = clf_dt_pruned_p.fit(X_p_train, y_p_train)

    predicted_t = ' '.join(map(str, clf_dt_pruned_t.predict([[length, weight]])))
    predicted_p = ' '.join(map(str, clf_dt_pruned_p.predict([[length, weight]])))
    score_t = clf_dt_pruned_t.score(X_test, y_test)*100
    score_p = clf_dt_pruned_p.score(X_p_test, y_p_test)*100

    print(f'\nDecisionTreeClassifier predicted you should wear t-shirt size: '
          f'{predicted_t} with a confidence of {round(score_t, 2)}%')
    print(f'DecisionTreeClassifier predicted you should wear pant size: '
          f'{predicted_p} with a confidence of {round(score_p, 2)}%')


def main():
    male_data = pd.read_csv('cleaned_male.csv')
    female_data = pd.read_csv('cleaned_female.csv')

    sex = input("Male or female? ")
    weight = input("Your weight in kg: ")
    length = input("Your height in cm: ")

    weight = int(weight) * 10
    length = int(length) * 10

    if sex.lower() == 'male':
        sex = male_data
        add_t_size_male(sex)
        add_p_size_male(sex)
    else:
        sex = female_data
        add_t_size_female(sex)
        add_p_size_female(sex)

    neighbour(length, weight, sex)
    plot_graph(sex)
    best_fit = sex.sort_values(by=['K']).head(5)

    most_frequent_t = most_frequent(best_fit['Size_t'])
    most_frequent_p = most_frequent(best_fit['Size_p'])

    print(f'\nMy function predicted you should wear t-shirt size: {most_frequent_t}')
    print(f'My function predicted you should wear pant size: {most_frequent_p}')
    kneigbor_predict(sex, length, weight)
    dtree_predict(sex, length, weight)


if __name__ == '__main__':
    main()
