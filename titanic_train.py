from sklearn.model_selection import train_test_split,RandomizedSearchCV
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold,GenericUnivariateSelect,f_classif,SelectFromModel,RFE
from scipy.stats import uniform,randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
# 泰坦尼克号为分类问题，目标是预测乘客是否生还，0表示未生还，1表示生还
data = pd.read_csv("train.csv")
# print(data.info())
# print(data.isna().sum())
# print(data.head())

# 获取特征和目标
# 删除无关项
X = data.drop(["Survived","PassengerId","Cabin","Ticket","Name"],axis=1)
y = data["Survived"]

# 切割数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# 对数据y进行非数值处理（实际不需要）
label_y = LabelEncoder()
y_train_la = label_y.fit_transform(y_train)
y_test_la = label_y.transform(y_test)

# 处理X中的缺失值和非数值属性
num_cols = ["Age","SibSp","Parch","Fare"] # 表示连续数据属性
cat_cols_oh = ["Sex","Embarked"] # 表示非数值属性
cat_cols_ord = ["Pclass"] # 表示不连续（只有1，2，3），但是有大小关系的（cat 是分类的缩写）

num_pipeline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler()) # 实际上树模型不需要做标准化
    ]
)
cat_pipeline_oh = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
    ]
)
cat_pipeline_ord = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))
    ]
)

# 对不同的属性进行不同的处理
preprocessor = ColumnTransformer(
    transformers=[
        ("num_transformer",num_pipeline,num_cols),
        ("cat_transformer_oh",cat_pipeline_oh,cat_cols_oh),
        ("cat_transformer_ord",cat_pipeline_ord,cat_cols_ord)
    ]
)

# 进行特征工程的比对
model = XGBClassifier(random_state=42,objective="binary:logistic")
# 得到 var 的准确度最高
pipe_var = Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("filter",VarianceThreshold(threshold=0.01)),
        ("estimator",XGBClassifier(random_state=42,objective="binary:logistic"))
    ]
)

# pipe_gen = Pipeline(
#     steps=[
#         ("preprocessor",preprocessor),
#         ("feature_selection",GenericUnivariateSelect(score_func=f_classif,mode="k_best",param=5)),
#         ("estimator",XGBClassifier(random_state=42,objective="binary:logistic"))
#     ]
# )
# pipe_sfm = Pipeline(
#     steps=[
#         ("preprocessor",preprocessor),
#         ("feature_selection",SelectFromModel(XGBClassifier(random_state=42,objective="binary:logistic"))),
#         ("estimator",XGBClassifier(random_state=42,objective="binary:logistic"))
#     ]
# )
# pipe_rfe = Pipeline(
#     steps=[
#         ("preprocessor",preprocessor),
#         ("feature_selection",RFE(XGBClassifier(random_state=42,objective="binary:logistic"),n_features_to_select=5)),
#         ("estimator",XGBClassifier(random_state=42,objective="binary:logistic"))
#     ]
# )
#
# # 需要找到最好的特征工程
# pipe_list = [
#     ("var",pipe_var),
#     ("gen",pipe_gen),
#     ("sfm",pipe_sfm),
#     ("rfe",pipe_rfe)
# ]
#
# param_list = {
#     "estimator__max_depth": randint(2,7),
#     "estimator__learning_rate": uniform(0.01,0.49),
#     "estimator__n_estimators": randint(50,500),
#     "estimator__subsample": uniform(0.5,0.5),
#     "estimator__colsample_bytree": uniform(0.5,0.5),
#     "estimator__reg_alpha": uniform(0,1),
#     "estimator__reg_lambda": uniform(0,1)
# }
#
# best_score = 0
# best_name = ""
# best_pipe = None
# for name,pipe in pipe_list:
#     search = RandomizedSearchCV(
#         estimator = pipe,
#         param_distributions = param_list,
#         n_iter = 10,
#         cv = 5,
#         verbose = 1,
#         random_state = 42,
#     )
#     search.fit(X_train,y_train_la)
#     cur_score = search.best_score_
#     if cur_score > best_score:
#         best_score = cur_score
#         best_name = name
#         best_pipe = search.best_estimator_
#
# print(f"特征选择最优方案：{best_name}, CV AUC={best_score:.4f}")
# print(best_pipe)

# 进行精细化的网格搜索
param_list = {
    "estimator__max_depth": randint(2,10),
    "estimator__learning_rate": uniform(0.01,0.59),
    "estimator__n_estimators": randint(50,600),
    "estimator__subsample": uniform(0.3,0.6), # loc=0.3 scale=0.6 → 范围0.3~0.9
    "estimator__colsample_bytree": uniform(0.3,0.6),
    "estimator__reg_alpha": uniform(0,1),
    "estimator__reg_lambda": uniform(0,1)
}
search = RandomizedSearchCV(
    estimator = pipe_var,
    param_distributions = param_list,
    n_iter = 20,
    cv = 5,
    verbose = 1,
    random_state = 42,
)
search.fit(X_train,y_train_la)

print(search.best_score_) # 0.8146754653796908
print(search.best_estimator_)

# 随机森林
# pipe_rf = Pipeline(
#     steps=[
#         ("preprocessor",preprocessor),
#         ("filter",VarianceThreshold(threshold=0.01)),
#         ("estimator",RandomForestClassifier(random_state=42)) # 换成RF
#     ]
# )

# # 进行精细化的网格搜索
# param_list = {
#     "estimator__max_depth": randint(2,10),
#     "estimator__min_samples_split": randint(2,10),
#     "estimator__n_estimators": randint(50,600),
# }
#
# search = RandomizedSearchCV(
#     estimator = pipe_rf,
#     param_distributions = param_list,
#     n_iter = 20,
#     cv = 5,
#     verbose = 1,
#     random_state = 42,
# )
# search.fit(X_train,y_train_la)
#
# print(search.best_score_) # 0.8273416724120949
# print(search.best_estimator_)



# pipe_lr = Pipeline([
#     ("preprocessor", preprocessor),
#     ("filter", VarianceThreshold(threshold=0.01)),
#     ("estimator", LogisticRegression(random_state=42, max_iter=500))
# ])
#
# param_lr = {
#     "estimator__C": uniform(0.01, 10)
# }
#
# search = RandomizedSearchCV(
#     estimator = pipe_lr,
#     param_distributions = param_lr,
#     n_iter = 20,
#     cv = 5,
#     verbose = 1,
#     random_state = 42,
# )
# search.fit(X_train,y_train_la)
#
# print(search.best_score_) # 0.796405003447257
# print(search.best_estimator_)

import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]
best_pipe = search.best_estimator_
train_sizes, train_scores, cv_scores = learning_curve(
    best_pipe, X_train, y_train, cv=5, scoring="roc_auc",
    train_sizes=np.linspace(0.1,1.0,10), random_state=42, n_jobs=1
)
train_mean = np.mean(train_scores, axis=1)
cv_mean = np.mean(cv_scores, axis=1)

plt.plot(train_sizes, train_mean, "o-", label="train auc")
plt.plot(train_sizes, cv_mean, "o-", label="cv auc")
plt.legend()
plt.xlabel("训练样本数量")
plt.ylabel("AUC")
plt.title("学习曲线")
plt.show() # 存在轻微的过拟合


# 保存模型
joblib.dump(best_pipe, "titanic_best_pipeline.joblib")
print("模型保存完成")