from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def lr_model(X_train, y_train, X_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def svc_model(X_train, y_train, X_test):
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def dtree_model(X_train, y_train, X_test):
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def random_forest(X_train, y_train, X_test):
    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def ada_boost(X_train, y_train, X_test):
    ada_clf = AdaBoostClassifier( DecisionTreeClassifier(max_depth=1), n_estimators=500, algorithm="SAMME.R") 
    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)
    return y_pred

def grad_boost(X_train, y_train, X_test):
    gbrt = GradientBoostingRegressor(n_estimators=500) 
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_test)
    return y_pred

def metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return acc, prec, rec, f1

