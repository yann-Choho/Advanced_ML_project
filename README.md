# ENSAE Paris | Institut Polytechnique de Paris

## Advanced Machine Learning Project (3A)

<center><img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/LOGO-ENSAE.png/480px-LOGO-ENSAE.png"></center>

## Topic : Deep learning models robust to adversarial attacks

### Realised by : 

* Thomas GABORIEAU 
* Choho Yann Eric CHOHO
* Nicolas DRETTAKIS


### Teacher: 

* Austin STROMME

#### Academic year: 2023-2024

November 2023 - January 2024.


### GitHub repository description:

Deep neural networks such as CNN's are sensitive to adversarial attacks, that is they can classify a imperceptably perturbed input differently than the unperturbed one. The questions arose in the articles below are about the exsistence of models robust to such attacks. We would like to encode and train models which can resist different type of attacks such as Fast Gradient Sign Method or Projected Gradient Descent, and then compare our results with "naturally trained" networks. We would also like to test the method on different databases to understand possible empirical limits. 

### How to reproduce our results 

By running the notebook attack_defences.ipynb you will obtain every accuracy result we found. It starts by running the notebook Chinese_MNIST_preprocessing.ipynb which preprocesses the data from the dataset we chose, and then you have the training of the model before every attack then defense methods. At the end of each method there are performance tests and visualisation of the perturbation on the images. You can also find several other notebooks containing some methods alone. 
