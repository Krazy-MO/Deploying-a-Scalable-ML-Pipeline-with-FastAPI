## Model Details
Jonathon Mohon 9/22/2026

The model type: Random Forest Classifier from sklearn, with random state 42. It predicts whether a persons income is over or under $50,000 from census information.
## Intended Use

This model is for learning purposes within a project to demonstrate how to deploy a ML pipeline.

This is not intended for use in real-world applications.
## Training Data
The training data was obtained from the UCI Machine Learning Repository.
There are 32,561 observations and 14 features plus the salary label.
The data was split into 80% training and 20% testing.
Processing the eight categorical features was done using one-hot encoding. The label was binarized with LabelBinarizer.
## Evaluation Data
The 20% of the model never saw during training, processed with the same encoder.
## Metrics
The following metrics were used:
- Precision (0.7945): when the model predicts >$50,000, it's right 79% of the time.
- Recall (0.5893): of the people who earn >$50,000, the model finds 58% of them.
- F1 Score (0.6767): a single score to balance precision and recall.

Performance on individual groups is in 'slice_output.txt'.
## Ethical Considerations
The model uses sensitive attributes like race and gender as inputs.
The data is from 1994, reflecting on the wages and patterns of the time.
Performance is not equal across groups. The model misses more high earners in smaller groups.
## Caveats and Recommendations
Small slices give unreliable scores. The '?' values were treated as their own category and not handled separately. More and or newer data could improve the model.