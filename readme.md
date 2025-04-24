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

### Models Used:
- tabularisai/multilingual-sentiment-analysis
- LiYuan/amazon-review-sentiment-analysis
- DataMonke/bert-base-uncased-finetuned-review-sentiment-analysis

  
## Labels

Data is labelled on a scale of 1-5, similar to the star score used in the actual reviews. These real ratings, however, are not factored in the model.


## Methods

Labels are oversampled and stratified prior to being fed into a model. There are several models included in steps, both regressors and classifiers.
They can be plugged into or unplugged from the pipeline and evaluated on using the appropriate evaluation step.

## Results (Best)


### 10K Dataset (v2)

Classification (Random Forest):  
Vectorizer: spaCy  
Normalization: L2 Normalizer  
Balancing: Oversample  
[INFO] Best parameters from grid search: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}  
[INFO] Training complete for random_forest_classification.  
[INFO] Evaluating classification model...  
[INFO] Accuracy: 0.8804  
[INFO] F1 Score (macro): 0.8801  
  
  
Classification (Support Vector Machine):  
Vectorizer: spaCy  
Normalization: L2 Normalizer  
Balancing: Oversample  
[INFO] Best parameters from grid search: {'C': 100, 'gamma': 'scale', 'kernel': 'poly'}  
[INFO] Training complete for support_vector_classification.  
[INFO] Evaluating classification model...  
[INFO] Accuracy: 0.7348  
[INFO] F1 Score (macro): 0.7340  
  
  
Classification (Multinomial Naive Bayes)  
Vectorizer: BOW  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
[INFO] Best parameters from grid search: {'alpha': 0.1}  
[INFO] Training complete for multinomial_naive_bayes_classification.  
[INFO] Evaluating classification model...  
[INFO] Accuracy: 0.8399  
[INFO] F1 Score (macro): 0.8387  
  
  
Classification (Gaussian Naive Bayes)  
Vectorizer: BOW  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  
[INFO] Best parameters from grid search: {'var_smoothing': 1e-07}  
[INFO] Training complete for gauss_naive_bayes_classification.  
[INFO] Evaluating classification model...  
[INFO] Accuracy: 0.7607  
[INFO] F1 Score (macro): 0.7575  


### 22k Dataset (v3)
Classification (Random Forest)
Vectorizer: spaCy  
Normalization: L2 Normalizer  
Balancing: Oversample (30k total)  
[INFO] Best parameters from grid search: {'max_depth': 25, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}
[INFO] Training complete for random_forest_classification.
[INFO] Evaluating classification model...
[INFO] Accuracy: 0.9134
[INFO] F1 Score (macro): 0.9135


Classification (Gaussian Naive Bayes)  
Vectorizer: BOW  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  (30k total)  
[INFO] Best parameters from grid search: {'var_smoothing': 1e-07}
[INFO] Training complete for gauss_naive_bayes_classification.
[INFO] Evaluating classification model...
[INFO] Accuracy: 0.6342
[INFO] F1 Score (macro): 0.6327

Classification (Multinomial Naive Bayes)  
Vectorizer: BOW  
Normalization/Scaler: MinMax Scaling  
Balancing: Oversample  (30k total)  
[INFO] Best parameters from grid search: {'alpha': 0.1}  
[INFO] Training complete for multinomial_naive_bayes_classification.  
[INFO] Evaluating classification model...  
[INFO] Accuracy: 0.7991  
[INFO] F1 Score (macro): 0.7992  


Classification (Support Vector Machine):  
Vectorizer: spaCy  
Normalization: L2 Normalizer  
Balancing: Oversample  
[INFO] Best parameters from grid search: {'C': 200, 'gamma': 'scale', 'kernel': 'poly'}  
[INFO] Training complete for support_vector_classification.  
[INFO] Evaluating classification model...  
[INFO] Accuracy: 0.7406  
[INFO] F1 Score (macro): 0.7405  