# Feature-Selection-For-Time-Series

## Summary
In this project we analized nine time series' datasets.
We used:
  - TSFresh library to extract features from the time series;
  - MCFS, Feature Agglomeration and Correlation to select relevant feature and reduce the dimensionality of the dataset;
  - K-means to valutate feature selection using unsupervisioned classification;
  - Similarity Functional Dependency to discovere relaxed functional dependency.
  

## How to run

*Please note: feature extraction and datasets' splitting into train sets and test sets have been already made and the result has been saved into **Pickle** folder.* 

### Feature selection and k-means
Run **main.py** specifing, in the order, as arguments: the dataset's name and the number of feature to select.
  - The result of kmeans' execution (in terms of scores) will be saved into **result.log** that it is contained into **Logs** folder. 

### Similarity Functional Dependency and k-means
1. Run **similarity_functional_dependency.py**
    - It discovers similarity functional dependencies on the ten features extracted using MCFS and Correlation, for every dataset.
    - The discovered similarity functional dependencies will be shown into **discovered_rfd.log** that it is contained into **Logs** folder.
2. Run **clustering_using_rfd.py** specifing, in the order, as arguments:
    - The type of feature selection;
    - The name of the dataset;
    - One or more feature to delete from the initial ten features, according to the similarity functional dependencies that were discovered from the above-mentioned dataset and the above-mentioned feature selection's.
    - The result of kmeans' execution (in terms of scores) will be saved into **kmeans_rfd.log** that it is contained into **Logs** folder.
