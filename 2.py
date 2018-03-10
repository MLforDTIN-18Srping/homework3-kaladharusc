import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import weka.core.jvm as jvm
import itertools
##################################################### Common Functions #################################################
def read_data():
    train_data = pd.read_csv("aps_failure_training_set.csv", skip_blank_lines=False)
    test_data = pd.read_csv("aps_failure_test_set.csv", skip_blank_lines=False)

    return train_data,test_data

def plot_corr(correlations, columns):
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(columns), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)
    plt.xticks(rotation=90)
    savefig("./plots/2_b_iii_correlations.png")
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def plot_roc_curve(y_true, y_predict_scores, charname):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    #for i in range(n_classes):
    fpr[2], tpr[2], _ = roc_curve(y_true, y_predict_scores, pos_label="pos", drop_intermediate=False)
    roc_auc[2] = auc(fpr[2], tpr[2])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_predict_scores.ravel(), pos_label="pos")
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    savefig(charname)

##################################################### Common Functions #################################################
# =======================================================================================================================
def b_i_impute_data():
    train_data, test_data = read_data()
    train_data[train_data.columns] = train_data[train_data.columns].replace(to_replace='na', value=np.NaN)
    test_data[test_data.columns] = test_data[test_data.columns].replace(to_replace='na', value=np.NaN)
    test_data = test_data.apply(pd.to_numeric, errors='ignore')
    train_data = train_data.apply(pd.to_numeric, errors='ignore')
    train_data.fillna(train_data.mean(), inplace=True)
    test_data.fillna(test_data.mean(), inplace=True)
    # print(train_data.head(2).to_csv())
    # print(train_data.shape)
    return train_data, test_data
# ======================================================================================================================
def b_ii_coefficient_variation():
    train_data, test_data = b_i_impute_data()
    mean_values = train_data.mean()
    std_values = train_data.std()
    CV = std_values/mean_values
    CV = CV.sort_values(ascending=False)

    print(CV.to_csv())
    return CV
# ======================================================================================================================
def b_iii_correlation_matrix():
    train_data, test_data = b_i_impute_data()
    train_data = train_data.drop("class", axis=1)
    correlations = train_data.corr()
    plot_corr(correlations, train_data.columns.values)
# ======================================================================================================================
def b_iv_plots():
    train_data, test_data = b_i_impute_data()

# ======================================================================================================================
def b_v_numberof_pos_neg():
    train_data, test_data = b_i_impute_data()
    classes = train_data["class"]

    print(classes.value_counts())
# ======================================================================================================================


def c_random_forest():
    train_data, test_data = b_i_impute_data()
    y_train = train_data["class"]
    x_train = train_data.drop("class", axis=1)

    y_test = test_data["class"]
    x_test = test_data.drop("class", axis=1)

    classification_scores = []
    oob_errors = []
    n_componenets = range(1,100)
    for n in n_componenets:
        print(n)
        rfc = RandomForestClassifier(n_estimators=n, oob_score=True, n_jobs=4)
        rfc.fit(x_train, y_train)
        y_predict = rfc.predict(x_test)
        classification_scores.append(accuracy_score(y_test, y_predict))
        oob_errors.append(1-rfc.oob_score_)

    #plot oob errorplot
    min_score = min(oob_errors)
    min_index = oob_errors.index(min_score)
    print(oob_errors)
    plt.plot(n_componenets,oob_errors,label = "RandomForestClassifier, max_features='sqrt'")
    plt.plot(n_componenets[min_index],oob_errors[min_index],marker='X', ms=6)
    plt.legend(loc="upper right")
    savefig("./plots/2_c_oob_errors.png")
    plt.close()

    max_score = max(classification_scores)
    index = classification_scores.index(max_score)
    n_tree = index+1
    rfc = RandomForestClassifier(n_estimators=n_tree)
    rfc.fit(x_train, y_train)
    y_predict = rfc.predict(x_train)
    #confusion Matrix Train Data
    print("train accuracy score", accuracy_score(y_train, y_predict))
    cm = confusion_matrix(y_train, y_predict, labels=["neg", "pos"])

    plt.figure()
    plot_confusion_matrix(cm, classes=["neg", "pos"],
                          title='Confusion matrix, without normalization')
    savefig("./plots/2_c_train_confusion_matrix.png")
    plt.close()

    # confusion Matrix Test Data
    y_predict = rfc.predict(x_test)
    print("test accuracy score",accuracy_score(y_test, y_predict))

    cm = confusion_matrix(y_test, y_predict, labels=["neg", "pos"])
    plt.figure()
    plot_confusion_matrix(cm, classes=["neg", "pos"],
                          title='Confusion matrix, without normalization')
    savefig("./plots/2_c_test_confusion_matrix.png")
    plt.close()

    #ROC curve Train Data
    Y_score = rfc.predict_proba(x_train)[:,1:]
    plot_roc_curve(y_train,Y_score,"./plots/2_c_train_ROC_curve.png")
    plt.close()

    # ROC curve Test Data
    Y_score = rfc.predict_proba(x_test)[:, 1:]
    plot_roc_curve(y_test, Y_score, "./plots/2_c_test_ROC_curve.png")
    plt.close()
# ======================================================================================================================

def savefig(*args):
    plt.savefig(bbox_inches='tight', *args)
def d_random_forest_balanced():
    train_data, test_data = b_i_impute_data()
    y_train = train_data["class"]
    x_train = train_data.drop("class", axis=1)

    y_test = test_data["class"]
    x_test = test_data.drop("class", axis=1)

    classification_scores = []
    oob_errors = []
    n_componenets = range(1,100)
    for n in n_componenets:
        print(n)
        rfc = RandomForestClassifier(n_estimators=n, oob_score=True, n_jobs=4, class_weight="balanced")
        rfc.fit(x_train, y_train)
        y_predict = rfc.predict(x_test)
        classification_scores.append(accuracy_score(y_test, y_predict))
        oob_errors.append(1-rfc.oob_score_)

    #plot oob errorplot
    min_score = min(oob_errors)
    min_index = oob_errors.index(min_score)
    print(oob_errors)
    plt.plot(n_componenets,oob_errors,label = "RandomForestClassifier, max_features='sqrt'")
    plt.plot(n_componenets[min_index],oob_errors[min_index],marker='X', ms=6)
    plt.legend(loc="upper right")
    savefig("./plots/2_d_oob_errors.png")
    plt.close()

    max_score = max(classification_scores)
    index = classification_scores.index(max_score)
    n_tree = index+1
    rfc = RandomForestClassifier(n_estimators=n_tree, class_weight="balanced")
    rfc.fit(x_train, y_train)
    y_predict = rfc.predict(x_train)
    #confusion Matrix Train Data
    print("train accuracy score", accuracy_score(y_train, y_predict))
    cm = confusion_matrix(y_train, y_predict, labels=["neg", "pos"])

    plt.figure()
    plot_confusion_matrix(cm, classes=["neg", "pos"],
                          title='Confusion matrix, without normalization')
    savefig("./plots/2_d_train_confusion_matrix.png")
    plt.close()

    # confusion Matrix Test Data
    y_predict = rfc.predict(x_test)
    print("test accuracy score",accuracy_score(y_test, y_predict))

    cm = confusion_matrix(y_test, y_predict, labels=["neg", "pos"])
    plt.figure()
    plot_confusion_matrix(cm, classes=["neg", "pos"],
                          title='Confusion matrix, without normalization')
    savefig("./plots/2_d_test_confusion_matrix.png")
    plt.close()

    #ROC curve Train Data
    Y_score = rfc.predict_proba(x_train)[:,1:]
    plot_roc_curve(y_train,Y_score,"./plots/2_d_train_ROC_curve.png")
    plt.close()

    # ROC curve Test Data
    Y_score = rfc.predict_proba(x_test)[:, 1:]
    plot_roc_curve(y_test, Y_score, "./plots/2_d_test_ROC_curve.png")
    plt.close()


import weka.core.converters as converters
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
import weka.plot.classifiers as plcls
def e_model_tree():
    # train_data, test_data = b_i_impute_data()
    # train_data.to_csv("./train_data.csv", index=False)
    # test_data.to_csv("./test_data.csv",index=False)

    jvm.start()
    train_data = converters.load_any_file("train_data.csv")
    train_data.class_is_first()

    test_data = converters.load_any_file("test_data.csv")
    test_data.class_is_first()

    print("1")
    cls = Classifier(classname="weka.classifiers.trees.LMT")
    print("2")
    cls.build_classifier(train_data)
    print("3")
    evl = Evaluation(train_data)


    evl.crossvalidate_model(cls, train_data, 5, Random(1))
    print("Train Accuracy:", evl.percent_correct)
    print("Train summary")
    print(evl.summary())
    print("Train class details")
    print(evl.class_details())
    print("Train confusion matrix")
    print(evl.confusion_matrix)
    plcls.plot_roc(evl, class_index=[0, 1], wait=True)
    savefig("./plots/e_train_roc_curve.png")

    evl = Evaluation(test_data)
    evl.test_model(cls, test_data)
    print("Test Accuracy:", evl.percent_correct)
    print("Test summary")
    print(evl.summary())
    print(" Testclass details")
    print(evl.class_details())
    print("Testconfusion matrix")
    print(evl.confusion_matrix)
    plcls.plot_roc(evl, class_index=[0, 1], wait=True)
    savefig("./plots/e_test_roc_curve.png")

    # train_data, test_data = b_i_impute_data()
    # y_train = train_data["class"]
    # x_train = train_data.drop("class", axis=1)
    #
    # y_test = test_data["class"]
    # x_test = test_data.drop("class", axis=1)
    #




if __name__ == '__main__':
    #b_i_impute_data()
    #b_ii_coefficient_variation()
    #b_iii_correlation_matrix()
    #b_v_numberof_pos_neg()
    #c_random_forest()
    d_random_forest_balanced()
    #e_model_tree()