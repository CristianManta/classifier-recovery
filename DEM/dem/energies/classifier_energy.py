from typing import Optional

import matplotlib.pyplot as plt
import torch
from fab.utils.plotting import plot_contours, plot_marginal_pair
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.logging_utils import fig_to_image

from utils.main import ModifiedResNet18, CustomCNN10, GMMMLP


class Classifier(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        n_mixes=40,
        loc_scaling=40,
        log_var_scaling=1.0,
        device="cpu",
        true_expectation_estimation_n_samples=int(1e5),
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=50,
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
        cls=0, # index of the class that we want to sample from
        resize_shape=10, # Size to rescale the images to
        energy_type='log_prob' # Choices: 'log_prob' or 'logit'
    ):
        use_gpu = device != "cpu"
        torch.manual_seed(0)

        self.curr_epoch = 0
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.should_unnormalize = should_unnormalize
        self.data_normalization_factor = data_normalization_factor

        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size

        self.name = "classifier"
        self.cls = cls
        self.resize_shape = resize_shape
        self.energy_type = energy_type
        
        self.classifier = GMMMLP()
        self.classifier.load_state_dict(torch.load("GMMMLP-weights-isotropic.pth"))
        self.classifier.to(self.device)
        self.classifier.eval()

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )
    def dummy_energy(self, samples): # Used for debugging
        return torch.zeros(samples.shape[0], device=self.device, dtype=torch.float32)

    def setup_test_set(self):
        return torch.rand(1000, self.dimensionality, device=self.device) # shape: (1000, dimensionality), dtype=torch.float32, device=self.device

    def setup_train_set(self):
        train_samples = torch.rand(self.train_set_size, self.dimensionality, device=self.device)
        return self.normalize(train_samples)

    def setup_val_set(self):
        val_samples = torch.rand(self.val_set_size, self.dimensionality, device=self.device)
        return val_samples # shape: (2000, dimensionality), dtype=torch.float32, device=self.device

    def __call__(self, samples: torch.Tensor) -> torch.Tensor: # shape: (num_estimator_mc_samples, dimensionality)
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        with torch.no_grad():
            if self.energy_type == 'logit':
                energy = self.classifier(samples)[:, self.cls]
            elif self.energy_type == 'log_prob':
                all_logits = self.classifier(samples)
                energy = all_logits[:, self.cls] - torch.logsumexp(all_logits, dim=1)
            
        return energy # shape: (num_estimator_mc_samples,)


    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        self.curr_epoch += 1

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
        should_unnormalize: bool = False,
    ) -> None:
        if wandb_logger is None:
            return

        if self.should_unnormalize and should_unnormalize:
            samples = self.unnormalize(samples)
        samples_fig = self.get_single_dataset_fig(samples, name)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def get_single_dataset_fig(self, samples, name, plotting_bounds=(-1.4 * 40, 1.4 * 40)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        self.gmm.to("cpu")
        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=ax,
            n_contour_levels=50,
            grid_width_n_points=200,
        )

        plot_marginal_pair(samples, ax=ax, bounds=plotting_bounds)
        ax.set_title(f"{name}")

        self.gmm.to(self.device)

        return fig_to_image(fig)

    def get_dataset_fig(self, samples, gen_samples=None, plotting_bounds=(-1.4 * 40, 1.4 * 40)):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        self.gmm.to("cpu")
        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=axs[0],
            n_contour_levels=50,
            grid_width_n_points=200,
        )

        # plot dataset samples
        plot_marginal_pair(samples, ax=axs[0], bounds=plotting_bounds)
        axs[0].set_title("Buffer")

        if gen_samples is not None:
            plot_contours(
                self.gmm.log_prob,
                bounds=plotting_bounds,
                ax=axs[1],
                n_contour_levels=50,
                grid_width_n_points=200,
            )
            # plot generated samples
            plot_marginal_pair(gen_samples, ax=axs[1], bounds=plotting_bounds)
            axs[1].set_title("Generated samples")

        # delete subplot
        else:
            fig.delaxes(axs[1])

        self.gmm.to(self.device)

        return fig_to_image(fig)
