import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import xgboost
from sklearn.model_selection import KFold
##################################################### Common Functions #################################################
plt.style.use('ggplot')
not_predictors = ["state", "county", "community", "communityname"]
response_var = "ViolentCrimesPerPop"
def plot_corr(correlations, columns, name):
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
    plt.savefig(name)
def scatterplot(X, Y, xLabel, yLabel, xticks=None, yticks=None):
    plt.scatter(X, Y, s=2)
    plt.xlabel(xLabel)
    if xticks is not None:
        plt.xticks(np.arange(xticks.size), xticks)
    if yticks is not None:
        plt.yticks(np.arange(yticks.size), yticks)
    plt.ylabel(yLabel)

##################################################### Common Functions #################################################
#=======================================================================================================================
def a_read_data():
    data_df = pd.read_csv("communities.data", skip_blank_lines=False)
    print(data_df.shape)
    print(data_df.head(2).to_csv())
#=======================================================================================================================
def b_fill_nan():
    data_df = pd.read_csv("communities.data", skip_blank_lines=False)
    data_df[data_df.columns] = data_df[data_df.columns].replace(to_replace='?', value=np.NaN)
    data_df = data_df.apply(pd.to_numeric, errors='ignore')
    data_df.fillna(data_df.mean(), inplace=True)
    #print(data_df.isnull().sum())
    global not_predictors
    data_df = data_df.drop(not_predictors, axis=1)
    # print(data_df.shape)
    # print(data_df.head(2).to_csv())
    return data_df[:1495],data_df[1495:], data_df
#=======================================================================================================================
def c_correlation_matrix():
    train_data, test_data,_ = b_fill_nan()
    correlations = train_data.corr()
    plot_corr(correlations, train_data.columns.values, "./plots/1_c_correlations.png")
#=======================================================================================================================
def d_coefficient_variation():
    train_data, test_data,_ = b_fill_nan()
    mean_values = train_data.mean()
    std_values = train_data.std()
    CV = std_values/mean_values
    CV = CV.sort_values(ascending=False)

    print(CV.to_csv())
    return CV
#=======================================================================================================================
def e_plots():
    CV_series = d_coefficient_variation()
    train_data, test_data,_ = b_fill_nan()
    top_cvs = CV_series.head(11)
    Y = train_data[response_var]
    fig = plt.figure(figsize=(20, 10))
    box_plot_data = train_data[top_cvs.index]
    plt.boxplot(box_plot_data.as_matrix(), labels=top_cvs.index.values)
    plt.savefig("./plots/e_boxplot.png")
    plt.close()

    fig = plt.figure(figsize=(15, 20))
    i = 1
    for (key,value) in top_cvs.items():
        fig.add_subplot(5,2,i)
        X = train_data[key]
        scatterplot(X,Y, key, response_var)
        i += 1
    plt.savefig("./plots/e_scatter_plots.png")
    plt.close()
#=======================================================================================================================
def f_linear_regression():
    train_data, test_data,_ = b_fill_nan()

    y_train = train_data[response_var]
    x_train = train_data.drop(response_var, axis=1)

    y_test = test_data[response_var]
    x_test = test_data.drop(response_var,axis=1)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)

    mse = mean_squared_error(y_test, y_predict)
    print(mse)

def g_ridge_regression():
    train_data, test_data,_ = b_fill_nan()

    y_train = train_data[response_var]
    x_train = train_data.drop(response_var, axis=1)

    y_test = test_data[response_var]
    x_test = test_data.drop(response_var,axis=1)

    ridge_regression = RidgeCV(alphas=[0.001,0.01,0.1,1.0,10.0],cv=None, store_cv_values=True)
    ridge_regression.fit(x_train, y_train)

    print("selected lambda", ridge_regression.alpha_)
    y_predict = ridge_regression.predict(x_test)

    mse = mean_squared_error(y_test, y_predict)
    print(mse)

def h_lasso_regresion():
    train_data, test_data, _ = b_fill_nan()

    y_train = train_data[response_var]
    x_train = train_data.drop(response_var, axis=1)

    y_test = test_data[response_var]
    x_test = test_data.drop(response_var, axis=1)

    alphas = [-0.0001,-0.001, -0.01, -0.1, -1.0,-10.0, 0.001, 0.01, 0.1, 1.0, 10.0]

    lasso_regression = LassoCV(alphas=alphas, cv=None,max_iter=100)
    lasso_regression.fit(x_train, y_train)

    print("selected lambda", lasso_regression.alpha_)
    coeff_df = pd.DataFrame([lasso_regression.coef_], columns=train_data.columns.values[:-1]).transpose()
    coeff_df = coeff_df[coeff_df[0] != 0.0]
    print(coeff_df)

    y_predict = lasso_regression.predict(x_test)
    print("mean squared error", mean_squared_error(y_test, y_predict))

    lasso_regression_norm = LassoCV(alphas=alphas, cv=None, max_iter=100, normalize=True)
    lasso_regression_norm.fit(x_train, y_train)

    print("selected lambda norm", lasso_regression_norm.alpha_)
    coeff_df = pd.DataFrame([lasso_regression_norm.coef_], columns=train_data.columns.values[:-1]).transpose()
    coeff_df = coeff_df[coeff_df[0] != 0.0]
    print(coeff_df)

    y_predict_nrom = lasso_regression_norm.predict(x_test)
    print("mean squared error norm", mean_squared_error(y_test, y_predict_nrom))

def i_pcr():
    train_data, test_data, _ = b_fill_nan()

    y_train = train_data[response_var]
    x_train = train_data.drop(response_var, axis=1)

    y_test = test_data[response_var]
    x_test = test_data.drop(response_var, axis=1)

    components = np.arange(1,min(train_data.shape), 2)


    cv_scores = []
    test_errors = []
    for n in components:
        pca = PCA(n_components=n)
        cv_scores.append(np.mean(cross_val_score(pca,x_train, y_train, cv=5)))
        x_pca = pca.fit_transform(x_train)
        x_test_pca = pca.fit_transform(x_test)

        lr = LinearRegression()
        lr.fit(x_pca, y_train)

        y_predict = lr.predict(x_test_pca)
        test_errors.append(mean_squared_error(y_test, y_predict))

    #print(cv_scores)
    max_val = max(cv_scores)
    index = cv_scores.index(max_val)

    # pca = PCA(n_components=components[index], svd_solver="full")
    # x_pca = pca.fit_transform(x_train)
    #
    # x_test_pca = pca.fit_transform(x_test)
    # lr = LinearRegression()
    # lr.fit(x_pca,y_train)

    #y_predict = lr.predict(x_test_pca)
    # print(mean_squared_error(y_test, y_predict))

    min_val = min(test_errors)
    index_te = test_errors.index(min_val)
    print(components[index_te], min_val)
    plt.plot(components, test_errors, linestyle='--', marker='o', ms=3)
    plt.xlabel("n components")
    plt.ylabel("MSE")
    # plt.plot(components[index], cv_scores[index],marker='X', ms=6)
    # print(components[index], cv_scores[index])
    plt.savefig("./i_pcr_mse.png")
    plt.close()

    plt.plot(components, cv_scores, linestyle='--', marker='o', ms=3)
    plt.plot(components[index], cv_scores[index],marker='X', ms=6)
    print(components[index], cv_scores[index])
    plt.xlabel("n components")
    plt.ylabel("CV scores")
    plt.savefig("./i_pcr_cv_score.png")



def j_xg_boost():
    train_data, test_data, _ = b_fill_nan()

    y_train = train_data[response_var]
    x_train = train_data.drop(response_var, axis=1)

    y_test = test_data[response_var]
    x_test = test_data.drop(response_var, axis=1)




    cv_scores = []
    cv_test_scores = []
    i = 3
    cv_train_scores = []
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2,0.3]
    n_trees = range(1,20,1)

    for n in n_trees:
        print(n)
        model = xgboost.XGBRegressor(max_depth=n, learning_rate=0.1)
        folds = KFold(n_splits=5,random_state=7)
        result = np.mean(cross_val_score(model, x_train, y_train, cv=folds))
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        y_train_predict = model.predict(x_train)
        cv_test_scores.append(mean_squared_error(y_test,y_predict))
        cv_train_scores.append(mean_squared_error(y_train, y_train_predict))
        cv_scores.append(result)

    cv_scores = np.array(cv_scores)
    cv_test_scores = np.array(cv_test_scores)

    plt.errorbar(alphas, cv_scores, label=str(i)+"CV train")
    # plt.errorbar(alphas, cv_test_scores, label=str(i)+" Mean Squared Test data")
    # plt.errorbar(alphas, cv_train_scores, label=str(i) + " Mean Squared Train data")

    plt.xlabel("regularization parameter")
    plt.legend(loc="upper right")
    plt.savefig("./plots/1_j_xgboost_alpha_reg_parameter_final.png")


if __name__ == '__main__':

    #a_read_data()
    #b_fill_nan()
    #c_correlation_matrix()
    d_coefficient_variation()
    #e_plots()
    #f_linear_regression()
    #g_ridge_regression()
    #h_lasso_regresion()
    #i_pcr()
    #j_xg_boost()






