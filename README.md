# Dassault-10-to-6-sensor

Dassault's open data: reproduce the performance of 10 sensors using only 6 sensors for actors.

Dassault has actors who must wear 10 different sensors on very specific areas on the human body, to record at the maximum precision their location in a 3-dimensional space. As they are heavy, Dassault would like to weaken this requirement to 6 sensors. However, they know they lose in precision, in that same 3-dimensional space.

The objective is to re-create as accurately the location of the 6 sensors in real time.

The reality shows this is not yet possible. The Pearson correlation coefficient of the model I used is about 0.35, which is far enough to be Top 2 in Dassault's open data competition. It requires to correct the column names provided and to use a fine-tuned Extreme Gradient Boosting model.

An auto-learner model is provided in case you need to get different results using another learning method (generalized linear model, decision-tree boosting).

# The code

The code used to create the Top 2 solution is provided here: https://github.com/Laurae2/Dassault-10-to-6-sensor/blob/master/Dassault.R
