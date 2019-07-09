# Progressive rock vs. everything else

## Layout and dependencies (as needed)

`data/`: Dataset creation scripts
- pytorch: https://pytorch.org/get-started/locally/
- librosa: `conda install -c conda-forge librosa`

## Methodology

### Fully Connected Neural Network
We developed a fully connected neural network to discriminate between progressive andnon-progressive rock.  The features used for this model include the average of the MFCCsfor each frame across time as well as the covariance of all MFCCs across time.  We used 20MFCCs per frame and thus 210 covariance values (since the covariance matrix is symmetric,take half of the matrix values).The model architecture included an input layer of size 230 nodes, two hidden layers ofsize 200 nodes and 50 nodes, respectively, and an output layer of one node.  Between eachlayer was a rectified linear unit activation function and the loss function used was the meansquared error.  The labels for progressive rock was arbitrarily chosen to be one and a negativeone for non-progressive rock.  To classify a song, the decision was if the network output valueless than or equal to 0, it was considered to be non-progressive rock, and progressive rockotherwise.  Some network auxiliary parameters included a learning rate of .001 and an Adamoptimizer.  The stopping criteria for the optimal network weights were set using the validation set provided.  The network stopped updating weights if the validation loss increased as thetraining  loss  decreased;  this  method  helps  minimize  overfitting  to  provide  optimal  modelgeneralization.

### Dimensionality Reduction (PCA)
One dimensionality reduction technique investigated was Principal Component Analysis(PCA). PCA reduces dimensionality by projecting to a lower subspace with maximum variance.  This procedure takes the correlated variables finds a smaller representation of them that are uncorrelated.

## Results

### Fully Connected Neural Network
The validation set accuracy was 72% and the test set accuracy was 55% (see figures 5and 6 below).  In both testing cases, the network performed well in deciding whether a songwas progressive rock but had difficulty labeling songs as a non-progressive rock. Confusion matrix results.

![GitHub Logo](/FCNN-confusionMatrix.png)
Figure 5:  Validation set confusion matrix of the model predictions after training on the fulltraining data set.
Figure 6: Test set confusion matrix of the model predictions after training on the full trainingdata set with validation set stopping criteria. 
### Principal Component Analysis
Given the 230-dimensional input feature vector, we reduced dimensionality to 200, 100,50,  20,  10,  and 5 dimensions and then tested each reduced feature vector.  We found thatreducing  the  dimensionality  from  230  to  any  of  the  previously  mentioned  dimensions  hadvirtually no effect on the model accuracy.  In Figure 7 the MFCC features do not have highseparability in three dimensions which can imply low model accuracy.
Figure 7:  Training test set reduced to three dimensions using PCA.
