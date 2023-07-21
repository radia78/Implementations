import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# the base trainer for future classes
class TrainerBase:
    def __init__(
        self,
        device: str,
        model: nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        save_every: int=25
    ):
        # the trainer setup
        self.device = device
        self.model = model.to(device)
        self.train_data = train_data
        self.n_samples = len(train_data)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every

    def _criterion(self, y_predicted, y_actual):
        raise NotImplementedError
    
    def _run_batch(self, source, target):
        raise NotImplementedError
    
    def _run_epoch(self, epoch):
        losses = []

        for batch in self.train_data:
            losses.append(self._run_batch(batch))
        
        print(f"Epoch {epoch} | Avg Model Loss: {sum(losses)/self.n_samples}")
    
    def _save_checkpoint(self, epoch: int, path: str):
        # save the model
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, path)
        print(f"Epoch {epoch} | Training checkpoint saved at {path}")
    
    def train(self, max_epochs: int, chkpt_path: str, save_img: bool=True):
        for epoch in range(max_epochs):
            self.model.train() # turn on training mode
            self._run_epoch(epoch)

            # save only every 'save_every'
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, chkpt_path, save_img)

# the trainer for VAE models
class TrainerVAE(TrainerBase):
    def __init__(
        self, 
        device: str, 
        model: nn.Module, 
        train_data: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        scheduler=None, 
        save_every: int = 25
    ):
        # inherit parent trainer arguments
        super().__init__(device, model, train_data, optimizer, scheduler, save_every)

    def _criterion(self, y_predicted, y_actual, mu, logvar, minibatch_weight):

        # the difference between the real data and the reconstructed data
        recons_loss = F.mse_loss(y_predicted, y_actual)

        # difference between distributions
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 -logvar.exp(), dim=1)) 

        return minibatch_weight * kld_loss + recons_loss
    
    def _run_batch(self, batch):
        # separate source from target
        source, _ = batch

        # compute minibatch weight
        minibatch_weight = source.shape[0]/self.n_samples
        
        self.optimizer.zero_grad(set_to_none=True) # zero out gradients
        source = source.flatten(start_dim=1).to(self.device) # send the data to the GPU

        recons, mu, logvar = self.model(source) # get the mean, variance, and reconstructed data
        model_loss = self._criterion(source, recons, mu, logvar, minibatch_weight) # compute the loss

        model_loss.backward()
        self.optimizer.step()
        self.scheduler.step() if self.scheduler is not None else None

        return model_loss
    
# the trainer for GAN models
class TrainerGAN(TrainerBase):
    def __init__(
        self, 
        device: str, 
        model: nn.Module,
        D_model: nn.Module, 
        train_data: DataLoader, 
        optimizer: torch.optim.Optimizer,
        D_optimizer: torch.optim.Optimizer,
        scheduler=None,
        D_scheduler=None,
        save_every: int = 25
    ):
        super().__init__(device, model, train_data, optimizer, scheduler, save_every)
        self.D_model = D_model.to(device)
        self.D_optimizer = D_optimizer
        self.D_scheduler = D_scheduler
    
    def _criterion(self, real_logits, real_labels, fake_logits, fake_labels):
        """
        Args:
            real_logits: The logit outputs from the discriminator given the data from the real distribution
            real_lables: Tensor of 1s that indicate that the logits are from the real distribution
            fake_logits: The logit outputs from the discriminator given the data from the generator
            fake_labels: Tensor of 0s that indicate that the logits are from the fake distribution
        """
        real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels, reduction='mean') # loss on the real inputs
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels, reduction='mean') # loss on the fake inputs

        return real_loss + fake_loss
    
    def _D_train(self, source):
        
        # clear out the gradients
        self.D_optimizer.zero_grad(set_to_none=True)

        # load the dummy data
        fake_inputs = torch.randn(source.shape[0], 100, device=self.device)
        real_labels = torch.ones(source.shape[0], 1, device=self.device)

        # train the discriminator
        G_fake_output = self.model(fake_inputs) # data generated from generator
        D_fake_output = self.D_model(G_fake_output) # fake logits from discriminator
        D_real_output = self.D_model(source) # real logits from discriminator
        D_loss = self._criterion(
            D_real_output,
            real_labels,
            D_fake_output,
            torch.zeros(source.shape[0], 1, device=self.device)
        )
        D_loss.backward()
        self.D_optimizer.step()
        self.D_scheduler.step() if self.D_scheduler is not None else None
    
    def _G_train(self, source):
        # clear out the gradients
        self.optimizer.zero_grad(set_to_none=True)

        # load the dummy data
        fake_inputs = torch.randn(source.shape[0], 100, device=self.device)
        real_labels = torch.ones(source.shape[0], 1, device=self.device)

        # train the discriminator
        D_fake_output = self.D_model(self.model(fake_inputs)) # fake logits from discriminator

        # compute generator loss
        G_loss = F.binary_cross_entropy_with_logits(D_fake_output, real_labels)
        G_loss.backward()
        self.optimizer.step()
        self.scheduler.step() if self.scheduler is not None else None

        return G_loss.data.item()
    
    def _run_batch(self, batch):
        source, _ = batch
        source = source.to(self.device) # send the data to the GPU

        # run through the discriminator update
        self._D_train(source)

        # run through the generator update
        model_loss = self._G_train(source)

        return model_loss
