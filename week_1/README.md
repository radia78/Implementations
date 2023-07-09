# Generative Adverserial Network
Generative Adverserial Networks are basically two neural networks that competes against each other. The first network is called the 'Generator' and it's purpose is to generate data that are similar to the real data inputs. The second network is called the 'Discriminator' and it's purpose is to classify if the inputs it gets came from the real distribution or an output from the Generator. In theory, the best generator is the one that can completely fool the discriminator to the point that the Discriminator can only guess right half of the time.

## Implementation Notes
In this implementation, the Generator network is a simple forward Neural Network with leaky ReLU activation and a final tanh layer. The Discriminator is also a simple forward Neureal Network with leaky ReLU activation, but this time there's an addition of dropout after the activation layer.

## Results
After training for 100 epochs, here are the results:

After 25 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/gen_img_25.png" alt="Generated Image after 25 Epochs" width="100" height="100"/>

After 50 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/gen_img_50.png" alt="Generated Image after 50 Epochs" width="100" height="100"/>

After 75 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/gen_img_75.png" alt="Generated Image after 75 Epochs" width="100" height="100"/>

After 100 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/gen_img_100.png" alt="Generated Image after 100 Epochs" width="100" height="100"/>

It looks like that the image is shifting from a 0 to an 8 and then to a 5 for the same input.
