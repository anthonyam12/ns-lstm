# Heteroscedastic ANN
An ensemble method where each ANN is trained on a subset of the data. The prediction is based on each networks (weighted) predictions.

--------

# Description

The main idea of H-ANN, is the local variance over certain time periods will be captured by training a network in different locales. This will create an ensemble of *n* networks all trained with on a different locale with (potentially) a different local variance. 

The ensemble alone should provide better predictions than regular neural networks and other econometric models. One expansion would be to weight the predictions of the neural networks based on the variance of the time period they model. For instance, if each network is trained over some 30-day period, the variance of the values over that period will be computed. This variance will be tied to the neural network that was trained. In the prediction process a **GARCH** model is used to track the current measure of local variance. ANN's with variance measures close to the current local variance are weighted higher than the networks with more different variances. 

--------

# Relevant Materials 

Below is a list of papers and other material relevant to this project/research.

+ Robust maximum likelihood training of heteroscedastic probabilistic
neural networks; Zheng Rong Yanga, Sheng Chenb; Journal: Neural Networks; Year: 1998; Found at: https://eprints.soton.ac.uk/251026/1/nn-els98.pdf

+ 

Search Terms

+ heteroscedastic neural network; scholar.google.com
