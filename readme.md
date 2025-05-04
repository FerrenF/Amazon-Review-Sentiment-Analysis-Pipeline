## Data Science Capstone

Dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
<br/>
Tools: Anaconda, Jupyter
<br/>
Libraries: Pandas, spaCy, pyarrow (Included with pandas), sweetviz


### Environment
Anaconda is required for this project to function. Theoretically, you could install these packages with pip only,
but it probably won't work right.

#### Anaconda
Included in the root of the project is an anaconda environment .yml file.
This was exported using the command `conda env export > environment.yml`.

You can *import* this environment by creating a new conda environment while specifying it
as the source `conda env -f environment.yml` and then activating it.

#### Pip
For requirements that aren't included in the anaconda environment, you can install the rest of the
necessary packages by using pip's `requirement.txt` file - also located in the root of this project.
**Always install through anaconda first and THEN pip.**


## Dataset

We are using the unlabelled dataset found here: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
From this dataset, we pull 10,000 samples. 5000 of these samples are unfiltered, and because of the high ratio of good ratings to bad,
5000 of these samples are filtered to be taken only from reviews whose score is '3 stars' or less.
<br/>
<br/>
Originally, we hand-annotated. This took too much time, so we came up with a new method: multi-model pre-predictions.
We have a pre-trained, much larger and more complex model (e.g. BERT) make classifications of the text beforehand. We
reduce the likelihood of innaccurate predictions by taking the output betwen multiple models and compare them for 'agreements' on data points.
(If both models think a particular text is rated 5 for very positve, then we keep it. If there is a minor disagreement, such as one model predicting 4 and the other 5, then we take the higher or 'stronger' score. Otherwise, the point is discarded.)
<br/>
<br/>
Using this method, we increased the original 1000 data points to 4200 data points out of 10000 pulled. I repeat this process
for dataset v3 with 21,000 data points out of 50000 used.

### Models Used to autoclassify:
- tabularisai/multilingual-sentiment-analysis
- LiYuan/amazon-review-sentiment-analysis
- DataMonke/bert-base-uncased-finetuned-review-sentiment-analysis

  
## Labels

Data is labelled on a scale of 1-5, similar to the star score used in the actual reviews. A label of '1' would represent a very negative sentiment. Conversely, a label of '5' would represent a very positive sentiment.
These labels are not the same as the ratings (e.g. 1-star or 5-star) given in the dataset, and these ratings are not factored into the model.

#### Label Distribution
The balance of labels in the dataset is not distributed equally. To make them such, oversampling or undersampling is performed. The evaluations below all had their model use oversampling to achieve distributive balance.

2025-05-04 03:50:51,916 [INFO] Label counts before balancing: 
y_train
5    4012
1    1842
3     515
Name: count, dtype: int64
2025-05-04 03:50:55,141 [INFO] Label balancing complete using oversample method.
2025-05-04 03:50:55,142 [INFO] Label counts after balancing: 
y_train
1    2500
3    2500
5    2500	


## Methods

Labels are oversampled and stratified prior to being fed into a model. There are several models included in steps, both regressors and classifiers.
They can be plugged into or unplugged from the pipeline and evaluated on using the appropriate evaluation step.


## Results (Visual)

![F1 Score](f1_score_comparison.png)
![Accuracy](accuracy_comparison.png)
![Confusion Matrix](confusion.png)
![Wordcloud (Very Negative Sentiment)](wordcloud_1.png)
![Wordcloud (Very Positive Sentiment)](wordcloud_5.png)

## Results (Data)


## Dataset (V2)
Classification (Logistic Regression)  
Vectorizer: TF-IDF     
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best parameters from grid search: {'C': 10, 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga'}  
Training complete for 
logistic_regression.
Evaluating classification model...
Accuracy: 0.7219
Precision (macro): 0.5445
Recall (macro): 0.5445
F1 Score (macro): 0.5439

Classification (Logistic Regression)  
Vectorizer: TF-IDF with n gram 1-2 range     
Normalization/Scaler: L2 Normalizer 
Balancing: Oversample  
Best parameters from grid search: {'C': 10, 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga'}  
Training complete for 
logistic_regression.
Evaluating classification model...
Accuracy: 0.7269
Recall (macro): 0.5824
F1 Score (macro): 0.5762

Classification (Logistic Regression)  
Vectorizer: BOW     
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best parameters from grid search: {'C': 10, 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga'}  
Training complete for 
logistic_regression.
Evaluating classification model...
Accuracy: 0.7520
Precision (macro): 0.5960
Recall (macro): 0.5999
F1 Score (macro): 0.5976

Classification (Logistic Regression)  
Vectorizer: BOW     
Normalization/Scaler: L2 Normalizer  
Balancing: Oversample  
Best parameters from grid search: {'C': 10, 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga'}  
Training complete for 
logistic_regression.
Evaluating classification model...
Accuracy: 0.7232
Precision (macro): 0.5459
Recall (macro): 0.5448
F1 Score (macro): 0.5446

Classification (Logistic Regression)  
Vectorizer: spaCy     
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best parameters from grid search: {'C': 10, 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga'}  
Training complete for 
logistic_regression.
Evaluating classification model...
Accuracy: 0.5863
Precision (macro): 0.5026
Recall (macro): 0.5196
F1 Score (macro): 0.4872

Classification (Logistic Regression)  
Vectorizer: spaCy     
Normalization/Scaler: L2 Normalizer  
Balancing: Oversample  
Best parameters from grid search: {'C': 10, 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga'}  
Training complete for 
logistic_regression.
Evaluating classification model...
Accuracy: 0.7244
Precision (macro): 0.5469
Recall (macro): 0.5459
F1 Score (macro): 0.5457



Classification (Gaussian Naive Bayes)  
Vectorizer: TF-IDF 
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best Parameters: {'var_smoothing': 1e-08}  
Training complete for 
gauss_naive_bayes_classification.
Evaluating classification model...
Accuracy: 0.4375
Precision (macro): 0.4066
Recall (macro): 0.4000
F1 Score (macro): 0.3707 

Classification (Gaussian Naive Bayes)  
Vectorizer: TF-IDF 
Normalization/Scaler: L2 Normalizer  
Balancing: Oversample  
Best Parameters: {'var_smoothing': 1e-08}  
Training complete for 
gauss_naive_bayes_classification.
Evaluating classification model...
Accuracy: 0.4294
Precision (macro): 0.3916
Recall (macro): 0.3800
F1 Score (macro): 0.3588

Classification (Gaussian Naive Bayes)  
Vectorizer: spaCy 
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best Parameters: {'var_smoothing': 1e-08}  
Training complete for 
gauss_naive_bayes_classification.
Evaluating classification model...
Accuracy: 0.3622
Precision (macro): 0.4129
Recall (macro): 0.4109
F1 Score (macro): 0.3333

Classification (Gaussian Naive Bayes)  
Vectorizer: spaCy 
Normalization/Scaler: L2 Normalizer   
Balancing: Oversample  
Best Parameters: {'var_smoothing': 1e-08}  
Training complete for 
gauss_naive_bayes_classification.
Evaluating classification model...
Accuracy: 0.3622
Precision (macro): 0.4129
Recall (macro): 0.4109
F1 Score (macro): 0.3333

Classification (Gaussian Naive Bayes)  
Vectorizer: BOW 
Normalization/Scaler: MinMax Scaling 
Balancing: Oversample  
Best Parameters: {'var_smoothing': 1e-08}  
Training complete for 
gauss_naive_bayes_classification.
Evaluating classification model...
Accuracy: 0.4200
Precision (macro): 0.3958
Recall (macro): 0.3839
F1 Score (macro): 0.3561

Classification (Gaussian Naive Bayes)  
Vectorizer: BOW 
Normalization/Scaler: L2 Normalizer  
Balancing: Oversample  
Best Parameters: {'var_smoothing': 1e-08}  
Training complete for 
gauss_naive_bayes_classification.
Evaluating classification model...
Accuracy: 0.4407
Precision (macro): 0.4124
Recall (macro): 0.3999
F1 Score (macro): 0.3732

Classification (Multinomial Naive Bayes)  
Vectorizer: TF-IDF    
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best Parameters: {'alpha': 0.1}  
Training complete for 
multinomial_naive_bayes_classification.
Evaluating classification model...
Accuracy: 0.6591
Precision (macro): 0.5129
Recall (macro): 0.5234
F1 Score (macro): 0.5147

Classification (Multinomial Naive Bayes)  
Vectorizer: TF-IDF    
Normalization/Scaler: L2 Normalizer 
Balancing: Oversample  
Best Parameters: {'alpha': 0.1}  
Training complete for 
multinomial_naive_bayes_classification.
Evaluating classification model...
Accuracy: 0.6924
Precision (macro): 0.5493
Recall (macro): 0.5678
F1 Score (macro): 0.5531

Classification (Multinomial Naive Bayes)  
Vectorizer: BOW    
Normalization/Scaler: MinMax Scaling 
Balancing: Oversample  
Best Parameters: {'alpha': 0.1}  
Training complete for 
multinomial_naive_bayes_classification.
Evaluating classification model...
Accuracy: 0.6723
Precision (macro): 0.5244
Recall (macro): 0.5350
F1 Score (macro): 0.5268

Classification (Multinomial Naive Bayes)  
Vectorizer: BOW    
Normalization/Scaler: L2 Normalizer 
Balancing: Oversample  
Best Parameters: {'alpha': 0.1}  
Training complete for 
multinomial_naive_bayes_classification.
Evaluating classification model...
Accuracy: 0.7068
Precision (macro): 0.5668
Recall (macro): 0.5886
F1 Score (macro): 0.5709



SVM, Scaling, TF-IDF, No Igram  
Best parameters from grid search: {'C': 200, 'gamma': 'scale', 'kernel': 'rbf'}  
Training complete for support_vector_classification.  
Evaluating classification model...  
Accuracy: 0.6353  
Precision (macro): 0.7740  
Recall (macro): 0.3429  
F1 Score (macro): 0.2790  


SVM, Scaling, BOW, No Ingram  
Best parameters from grid search: {'C': 200, 'gamma': 'scale', 'kernel': 'rbf'}  
Training complete for support_vector_classification.  
Evaluating classification model...  
Accuracy: 0.6560  
Precision (macro): 0.5168  
Recall (macro): 0.5146  
F1 Score (macro): 0.5068  


RF, Scaling, BOW, No ingram  
Best parameters from grid search: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}  
Training complete for random_forest_classification.  
Evaluating classification model...  
Accuracy: 0.7100  
Precision (macro): 0.5554  
Recall (macro): 0.5384  
F1 Score (macro): 0.5292  


RF, Scaling, TF-IDF, No ingram  
Best parameters from grid search: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}  
Training complete for random_forest_classification.  
Evaluating classification model...  
Accuracy: 0.7144  
Precision (macro): 0.5552  
Recall (macro): 0.5433  
F1 Score (macro): 0.5367  


RF, Normalization, SpaCy, No ingram  
Best parameters from grid search: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}  
Training complete for random_forest_classification.  
Evaluating classification model...  
Accuracy: 0.6905  
Precision (macro): 0.5049  
Recall (macro): 0.4653  
F1 Score (macro): 0.4581  


KNN, Normalization, SpaCy, no Ingram  
Best parameters from grid search: {'metric': 'manhattan', 'n_neighbors': 2, 'weights': 'distance'}  
Training complete for k_nearest_neighbors_classification.  
Evaluating classification model...  
Accuracy: 0.4739  
Precision (macro): 0.3845  
Recall (macro): 0.3902  
F1 Score (macro): 0.3751  


KNN, Scaling, BOW, no Ingram  
Best parameters from grid search: {'metric': 'manhattan', 'n_neighbors': 2, 'weights': 'distance'}  
Training complete for k_nearest_neighbors_classification.  
Evaluating classification model...  
Accuracy: 0.5838  
Precision (macro): 0.4346  
Recall (macro): 0.4386  
F1 Score (macro): 0.4350  