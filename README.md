# Display_Model_History-
This repository provides a way to display training history for deep learning model in Keras.

When we build any Deep Learning Model, we always worry from **underfitting** or **overfitting** of our model. Also, to get the right model, 
you have to make sure that you do the right fitting for your model. 

1- Underfitting: 

- When a model cannot capture the underlying trend of the data, so underfitting has been occured. As result, The accuracy of the model is distroyed and the model is failed to 
fit thr data well enough. So, Underfitting: Poor performance on the training data and poor generalization to other data.
- Underfitting has low bias and high variance. 
- Underfitting can be avoided by using more data and also reducing the features by feature selection.

2- Overfitting:

- When we train a lot of data and fit it into our model and our model gets train too much by learning the noise. The model could reach to a level that it cannot categorize
data correctly bcause of too much noise.. here overfitting occurs. So, Overfitting: Good performance on the training data, poor generliazation to other data.
- Overfitting has high bias and low variance.

These following plots show the occurance of underfitting and overfitting in terms of the loss and number of epchos for both training and testing data set: 

![Under/Over fitting](https://github.com/omarnj-lab/Display_Model_History-/blob/main/overfit.jpg)

--------------------------------------------------------------
**The implementation**

Keras provides the capability to register callbacks when training a deep learning model. These callbacks record the process of training and testing and keep then saved
for further processing. We can use them to ensure that the model has not been failed for the two problems neither underfitting nor overfitting.

One of the default callbacks that is registered when training all deep learning models is the History callback. It records training metrics for each epoch.
This includes the loss and the accuracy of the model. Remember we must first fit te model to th validation sets to work well. 

You can call the history recoreds by the following code: 
...
# list all data in history
print(history.history.keys())
...

I applied this idea on a dataset and model that have been used in DL lesson.
Note that you can reach code in this file as well. 

THe following result have been shown up : 

For the accuracy of the model : 

![Model Accuracy](https://github.com/omarnj-lab/Display_Model_History-/blob/main/modelaccuracy.png)





From the plot of accuracy we can see that the model could probably be trained a little more as the trend for accuracy on both datasets is still rising for the last few epochs. We can also see that the model has not yet over-learned the training dataset, showing comparable skill on both datasets.


For the loss of the model : 




From the plot of loss, we can see that the model has comparable performance on both train and validation datasets (labeled test). If these parallel plots start to depart consistently, it might be a sign to stop training at an earlier epoch.

**Conclusion: 

By the using of the previous implementation we can evlaute our model and see if it is done a good fitting or under/over fitting
and based on that, we can judge the model and edit it if it is needed. 

