## Data Science Capstone
________________________

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
as the source `conda env -f encironment.yml` and then activating it.

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

2025-04-30 12:15:57,130 [INFO] Label counts before balancing: 
label
5    989
2    630
1    550
3    534
4    277
Name: count, dtype: int64

2025-04-30 12:15:56,817 [INFO] Train/test split complete. Train size: 2980, Test size: 746
2025-04-30 12:15:57,169 [INFO] Label counts after balancing: 
label
5    989
2    989
3    989
1    989
4    989

2025-04-29 23:00:23,656 [INFO] Train/test split complete. Train size: 15364, Test size: 3842
2025-04-29 23:00:26,132 [INFO] Label counts before balancing: 
label
5    5989
3    3139
2    2423
1    2227
4    1586
Name: count, dtype: int64
2025-04-29 23:00:26,309 [INFO] Label balancing complete using oversample method.
2025-04-29 23:00:26,312 [INFO] Label counts after balancing: 
label
2    5989
5    5989
4    5989
3    5989
1    5989
Name: count, dtype: int64


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


## 2.98K Dataset (V2) - Test Size 746


## 15.3k Dataset (V3)

2025-04-29 23:11:48,342 [INFO] Best parameters from grid search: {'max_depth': 25, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}
2025-04-29 23:11:48,342 [INFO] Training complete for random_forest_classification.
2025-04-29 23:11:48,823 [INFO] Evaluating classification model...
2025-04-29 23:11:49,942 [INFO] Accuracy: 0.5273
2025-04-29 23:11:49,959 [INFO] F1 Score (macro): 0.4055




###
### OLD DATA
###
### 4.2k Dataset (v2)

Classification (Random Forest):  
Vectorizer: spaCy  
Normalization: L2 Normalizer  
Balancing: Oversample  
Best Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}  
Model: random_forest_classification.
Accuracy: 0.8804  
F1 Score (macro): 0.8801  
  

Classification (Random Forest):  
Vectorizer: spaCy  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best Parameters: {'max_depth': 25, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}  
Model: random_forest_classification.
Accuracy: 0.8812  
F1 Score (macro): 0.8813  
  

Classification (Support Vector Machine):  
Vectorizer: spaCy  
Normalization: L2 Normalizer  
Balancing: Oversample  
Best Parameters: {'C': 100, 'gamma': 'scale', 'kernel': 'poly'}  
Model: support_vector_classification.
Accuracy: 0.7348  
F1 Score (macro): 0.7340    


Classification (Support Vector Machine):  
Vectorizer: spaCy  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best Parameters: {'C': 200, 'gamma': 'scale', 'kernel': 'poly'}  
Model: support_vector_classification.
Accuracy: 0.8464  
F1 Score (macro): 0.8463  
 

Classification (Multinomial Naive Bayes)  
Vectorizer: BOW  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best Parameters: {'alpha': 0.1}  
Model: multinomial_naive_bayes_classification.
Accuracy: 0.8399  
F1 Score (macro): 0.8387  


Classification (Multinomial Naive Bayes)  
Vectorizer: TF-IDF    
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best Parameters: {'alpha': 0.1}  
Model: multinomial_naive_bayes_classification.
Accuracy: 0.8391   
F1 Score (macro): 0.8375  


Classification (Gaussian Naive Bayes)  
Vectorizer: BOW  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best Parameters: {'var_smoothing': 1e-07}  
Model: gauss_naive_bayes_classification.  
Accuracy: 0.7607  
F1 Score (macro): 0.7575  


Classification (Gaussian Naive Bayes)  
Vectorizer: TF-IDF    
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best Parameters: {'var_smoothing': 1e-07}  
Model: gauss_naive_bayes_classification.  
Accuracy: 0.7437  
F1 Score (macro): 0.7392  


Classification (K-Nearest-Neighbors)  
Vectorizer: TF-IDF    
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best parameters from grid search: {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}  
Training complete for k_nearest_neighbors_classification.  
Evaluating classification model...  
Accuracy: 0.8230  
F1 Score (macro): 0.8240  


Classification (K-Nearest-Neighbors)  
Vectorizer: BOW     
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best parameters from grid search: {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}  
Training complete for k_nearest_neighbors_classification.  
Evaluating classification model...  
Accuracy: 0.8529  
F1 Score (macro): 0.8551  


Classification (K-Nearest-Neighbors)  
Vectorizer: BOW     
Normalization/Scaler: L2 Normalization  
Balancing: Oversample  
Best parameters from grid search:{'metric': 'euclidean', 'n_neighbors': 2, 'weights': 'distance'}   
Training complete for k_nearest_neighbors_classification.  
Evaluating classification model...  
Accuracy: 0.8367  
F1 Score (macro): 0.8358    


Classification (K-Nearest-Neighbors)  
Vectorizer: Spacy     
Normalization/Scaler: L2 Normalization  
Balancing: Oversample  
Best parameters from grid search:{'metric': 'manhattan', 'n_neighbors': 2, 'weights': 'distance'}   
Training complete for k_nearest_neighbors_classification.  
Evaluating classification model...  
Accuracy: 0.8383    
F1 Score (macro): 0.8372   


Classification (Logistic Regression)  
Vectorizer: BagOfWords     
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best parameters from grid search: {'C': 10, 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga'}  
Training complete for 
logistic_regression.
Evaluating classification model...  
Accuracy: 0.8925
F1 Score (macro): 0.8918

Classification (Logistic Regression)  
Vectorizer: TF-IDF     
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
Best parameters from grid search: {'C': 10, 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga'}  
Training complete for 
logistic_regression.
Evaluating classification model...  
Accuracy: 0.8901
F1 Score (macro): 0.8899



### 22k Dataset (v3)

Classification (Random Forest)  
Vectorizer: spaCy  
Normalization/Scaler: L2 Normalizer  
Balancing: Oversample (30k total)  
Best Parameters: {'max_depth': 25, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}  
Model: random_forest_classification.
Accuracy: 0.9134  
F1 Score (macro): 0.9135  


Classification (Random Forest)  
Vectorizer: spaCy  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample (30k total)  
Best Parameters: {'max_depth': 25, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}  
Model: random_forest_classification.
Accuracy: 0.9120  
F1 Score (macro): 0.9121  


Classification (Gaussian Naive Bayes)  
Vectorizer: BOW  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  (30k total)  
Best Parameters: {'var_smoothing': 1e-07}  
Model: gauss_naive_bayes_classification.
Accuracy: 0.6342  
F1 Score (macro): 0.6327  


Classification (Gaussian Naive Bayes)  
Vectorizer: TF-IDF  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  (30k total)  
Best Parameters: {'var_smoothing': 1e-07}  
Model: gauss_naive_bayes_classification.
Accuracy: 0.6772  
F1 Score (macro): 0.6712  


Classification (Multinomial Naive Bayes)  
Vectorizer: BOW  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  (30k total)  
Best Parameters: {'alpha': 0.1}  
Model: multinomial_naive_bayes_classification.
Accuracy: 0.7991  
F1 Score (macro): 0.7992  


Classification (Multinomial Naive Bayes)  
Vectorizer: TF-IDF  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  (30k total)  
Best Parameters: {'alpha': 0.1}  
Model: multinomial_naive_bayes_classification.
Accuracy: 0.8274  
F1 Score (macro): 0.8270  


Classification (Support Vector Machine):  
Vectorizer: spaCy  
Normalization/Scaler: L2 Normalizer  
Balancing: Oversample  
Best Parameters: {'C': 200, 'gamma': 'scale', 'kernel': 'poly'}  
Model: support_vector_classification.
Accuracy: 0.7406  
F1 Score (macro): 0.7405  


Classification (Support Vector Machine):  
Vectorizer: spaCy  
Normalization/Scaler: MinMax Scaler  
Balancing: Oversample  
Best Parameters: {'C': 200, 'gamma': 'scale', 'kernel': 'poly'}  
Model: support_vector_classification.
Accuracy: 0.9050  
F1 Score (macro): 0.9047  


Classification (K-Nearest-Neighbors)  
Vectorizer: Spacy     
Normalization/Scaler: L2 Normalization  
Balancing: Oversample  
Best parameters from grid search:{'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}     
Training complete for k_nearest_neighbors_classification.  
Evaluating classification model...  
Accuracy: 0.8748      
F1 Score (macro): 0.8736  


Classification (K-Nearest-Neighbors)  
Vectorizer: Spacy     
Normalization/Scaler: MinMax Scaler  
Balancing: Oversample  
Best parameters from grid search:{'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}     
Training complete for k_nearest_neighbors_classification.  
Evaluating classification model...  
Accuracy: 0.8743      
F1 Score (macro): 0.8730  


