
# SIGNET

This is my implementation of the [signet](https://arxiv.org/pdf/1707.02131.pdf) paper
which is a convolutional Siamese Network for writer independent offline signature
verification.



## Siamese Architecture

![App Screenshot](https://i.imgur.com/lwRkFYF.png)

Siamese networks are twin networks with shared weights, which can be trained to learn a feature space where similar observations
are placed in proximity. This is achieved by exposing the network to a pair of similar and dissimilar observations and minimizing the [Euclidean distance](http://mathonline.wikidot.com/the-distance-between-two-vectors) between similar pairs while simultaneously
maximizing it between dissimilar pairs.

```
def euclidian_distance(vectors):
    vect1 , vect2 = vectors
    sum_sq = K.sum(K.square(vect1 - vect2) , axis=1 , keepdims=True)
    dist = K.sqrt(K.maximum(sum_sq , K.epsilon()))
    return dist
```



## DATA PREPARATION

### CEDAR DATABASE

* The model is trained using the [cedar dataset](https://paperswithcode.com/dataset/cedar-signature) which can be downloaded from [here](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar).
* CEDAR signature database contains signatures of 55 signers.
* Each of these signers signed 24 genuine and 24 forged signatures.
* The dataset is resized to a fixed input size of 155 X 220 using [bilinear Interpolation](https://www.sciencedirect.com/topics/engineering/bilinear-interpolation) to train the network.
* The [cedar.py](https://github.com/Parijat-18/SigNet/blob/main/cedar.py) implements the preprocessing and division protocol as mentioned in the paper.



## Loss Function

[Contrastive loss](https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec) is the most commonly used loss function
for training siamese networks.It takes the output of the network for a positive example and 
calculates its distance to an example of the same class and contrasts that with 
the distance to negative examples.

```
def contrastive_loss_with_margin(margin): 
    def contrastive_loss(y_true, y_pred): 
        square_pred = K.square(y_pred) 
        margin_square = K.square(K.maximum(margin - y_pred, 0)) 
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square) 
    return contrastive_loss 
```



# RESULT

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
| Original | Genuine | Forged |
|<img src="https://github.com/Parijat-18/SigNet/blob/main/sample_imgs/org.jpg"> |<img 
src="https://github.com/Parijat-18/SigNet/blob/main/sample_imgs/org2.jpg">  |<img 
src="https://github.com/Parijat-18/SigNet/blob/main/sample_imgs/forg.jpg"> |
Predicted Label:| 0 => similar | 1 => forged |
