# Generative Adverserial Network
Generative Adverserial Networks are basically two neural networks that competes against each other. The first network is called the 'Generator' and it's purpose is to generate data that are similar to the real data inputs. The second network is called the 'Discriminator' and it's purpose is to classify if the inputs it gets came from the real distribution or an output from the Generator. In theory, the best generator is the one that can completely fool the discriminator to the point that the Discriminator can only guess right half of the time.

## Implementation Notes
In this implementation, the Generator network is a simple forward Neural Network with leaky ReLU activation and a final tanh layer. The Discriminator is also a simple forward Neureal Network with leaky ReLU activation, but this time there's an addition of dropout after the activation layer. The training comes down to feeding the Discriminator the real input and the generated input and compute a binary cross entropy loss to update the gradients of the Discriminator network while simultaneously computing the binary cross entropy loss between the Discriminator output for generated inputs and all 'true' labels to update the gradients of the Generator network.

## Results
After training for 100 epochs, here are the results:

#### After 25 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/GAN/gen_img_25.png" alt="Generated Image after 25 Epochs" width="100" height="100"/>

#### After 50 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/GAN/gen_img_50.png" alt="Generated Image after 50 Epochs" width="100" height="100"/>

#### After 75 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/GAN/gen_img_75.png" alt="Generated Image after 75 Epochs" width="100" height="100"/>

#### After 100 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/GAN/gen_img_100.png" alt="Generated Image after 100 Epochs" width="100" height="100"/>

It looks like that the image is shifting from a 0 to an 8 and then to a 5 for the same input.

# Variational Autoencoder
VAE is a neural network that attempts to mimic the suspected data generating process of a certain distribution by sampling data from the actual distribution and try to recreate a posterior and prior to a latent space data that has a mapping to the actual distribution.

## Implementation Notes
In this implementation, the VAE takes in a flattened image from the MNIST dataset and then encodes it using a few MLPs. It then outputs the mean and log variance of the image distribution. Afterwards, they sample some data from the latent distribution using the generated mean and log variance and then try to reconstruct the data. The decoder also uses a few MLPs to upsample from the latent space to reconstruct the flattened image.

## Results
After training for 100 epochs, here are the results:

#### After 25 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/VAE/recons_img_25.png" alt="Reconstructed Image after 25 Epochs" width="100" height="100"/>

#### After 50 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/VAE/recons_img_50.png" alt="Reconstructed Image after 50 Epochs" width="100" height="100"/>

#### After 75 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/VAE/recons_img_75.png" alt="Reconstructed Image after 75 Epochs" width="100" height="100"/>

#### After 100 Epochs
<img src="https://github.com/radia78/Text2Image/blob/main/week_1/VAE/recons_img_100.png" alt="Reconstructed Image after 100 Epochs" width="100" height="100"/>

It looks like that the reconstructed image from the VAE is much more smoother than the actual data distribution.
