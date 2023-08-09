# Sampahku
image-classification on type of waste <br />
Demo Sampahku app [click here](https://www.youtube.com/watch?v=hOKI60cqvV8&t=12s)

## Depedencies
* tensorflow 2.9.1
* python 3.10

## Dataset
For the dataset we use [These Github](https://github.com/sarahmfrost/compostnet) dataset from Sarah Frost, Bryan Tor, Rakshit Agrawal, Angus G. Forbes use for compostnet. The original dataset contain 7 class of waste type glass, paper, cardboard, plastic, metal, trash with contain 2.751 photo of waste in total. But for our dataset we deleted the trash class so the final class that we use for this project is:

* cardboard
* compost
* glass
* metal
* paper
* plastic

## Architecture
For our model we use transfer learning method for this project, We use VGG16 architecture which is a convolutional neural network that is 16 layers deep and we add a few more layers on top of that, you can see the detail about our model below

![final_model_plot](https://user-images.githubusercontent.com/73216938/173172380-016b6b68-0cf5-48d0-8b8b-952657bd8d32.png)

## Result
### accuracy and validation accuracy
we achieve pretty good number consistenly 90% training accuracy with 80% accuracy on validation

<img width="467" alt="training and validation accuracy" src="https://user-images.githubusercontent.com/73216938/173172511-8e5f30fb-a1f8-41a7-8584-8ea93db06824.png">

### accuracy loss and validation loss
our losses also is preety low

<img width="470" alt="training and validation loss" src="https://user-images.githubusercontent.com/73216938/173172521-b8161c04-0102-4d08-acfd-bbaa2a92ed5a.png">
