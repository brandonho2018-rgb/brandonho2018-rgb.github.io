<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Quality of Portuguese "Vinho Verde" Redwine Classification

I applied machine learning techniques to predict the quality of "Vinho Verde" redwine from the UCI Machine Learning Repository (Cortez et al. 2019).
***

## Introduction 

Wine has become a widely enjoyed drink, consumed by many across the world for leisure or for gatherings. With about 870 million gallons of wine being consumed by the US alone in 2024, it is clear that there is a need for automated wine quality predictions rather than subjective wine tastings from human tasters (bw166/Gomberg, Fredrikson et al.). According to an article from the Gaurdian, on average only 10% of wine tasters were consistent with their tastings per year (Derbyshire, David 2013). To keep wine tasting objective and consistent, machine learning may be the answer to predicting wine quality.

I used the Wine Quality dataset from the UCI Machine Learning Repository, specifically only the red wine dataset. The intention of this project is to apply supervised classification machine learning models in order to predict the wine quality of redwine, specifically Portoguese "Vinho Verde" redwine.

## Data

The redwine dataset consists of 12 columns:
  1. fixed acidity
  2. volatile acidity
  3. citric acid
  4. residual sugar
  5. chlorides
  6. free sulfur dioxide
  7. total sulfur dioxide
  8. density
  9. pH
  10. sulphates
  11. alcohol
  12. quality
Our traget variable is 'quality' since this is the variable we would like to predict using the data from the other columns. Therefore we have 11 features to consider in this dataset. I chose to do binary classification rather than multiclass classification in order to simplify the 'quality' dataset, although multiclass classification may provide better results due to the skewing of the data as shown in figure 1.

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

<p>
When \(a \ne 0\), there are two solutions to \(ax^2 + bx + c = 0\) and they are
  \[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]
</p>

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)

