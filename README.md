# AllState-Kaggle
Files related to AllState Kaggle Competition


All State Insurance Claims Prediction is a Kaggle Competition Held in Winter 2016. The attached scripts were used to generate both
individual and ensemble models that were used for submission.

Target Variable = Claim Severity or losses (continuous)

Independent Variables = 132 completely anonymized features, including 14 continuous variables

Models Used - Xgboost and Deep Neural Network Ensembles

Xgboost model was used as a first layer model which was then blended into 5 other Deep Neural Network models in the second layer.

Rank Obtained - 685/3300 (Top 21%)

Ranking can be significantly improved by further tuning and cross validations. However, due to limitations in computing power
it was not attempted
