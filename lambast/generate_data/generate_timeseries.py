"""
Time series generation classes.

"""

import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import truncnorm
from typing_extensions import Self

from lambast.utils import assert_valid_covariance

from .timeseries_class import TimeSeries


class LinearSSM(TimeSeries):

    def __init__(self, state_matrix: NDArray, state_noise_cov: NDArray,
                 obs_matrix: NDArray, obs_noise_cov: NDArray,
                 rng: np.random.Generator | None = None,
                 scale_matrix: bool = False) -> None:
        """
        Initializes a new instance of a Linear State Space Model.
        State dimension is d, observation dimension is p.
        Assume zero-mean Gaussian noise in state and observation space.

        Parameters:
            state_matrix: shape (d,d) numpy array defining state
                transition matrix
            state_noise_cov: shape (d,d) numpy array defining
                state noise covariance
            obs_matrix: shape (p,d) numpy array defining
                observation matrix
            obs_noise_cov: shape (p,p) numpy array defininig
                observation noise covariance
            rng: numpy rng (else use default); see
                https://numpy.org/doc/2.0/reference/random/index.html#random-quick-start
            scale_matrix (bool): Whether to rescale the matrix for stability,
                default, false.
        """
        super().__init__()

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.d = state_matrix.shape[0]
        self.p = obs_matrix.shape[0]

        self.state_matrix = state_matrix
        self.state_noise_cov = state_noise_cov
        self.obs_matrix = obs_matrix
        self.obs_noise_cov = obs_noise_cov

        # Check if state matrix is stable and warn user if not, but only if
        # scale_matrix is false, if scale_matrix is true, just scale the matrix
        self.rescale_matrix(scale_matrix)

        # Check if covariance matrices are valid
        assert_valid_covariance(state_noise_cov)
        assert_valid_covariance(obs_noise_cov)

        # Check shapes
        if state_matrix.shape[0] != state_matrix.shape[1]:
            raise ValueError("State matrix is not square")
        if obs_matrix.shape[1] != self.d:
            raise ValueError("Obs matrix is not (p,d)")
        if state_noise_cov.shape[0] != self.d:
            raise ValueError("State cov matrix is not (d,d)")
        if obs_noise_cov.shape[0] != self.p:
            raise ValueError("Obs cov matrix is not (d,d)")

    def rescale_matrix(self, scale_matrix: bool) -> None:
        """
        Function that re-scales the state matrix to make it stable

        Parameter:
            scale_matrix: Whether to rescale the state matrix
        """
        eigenvalues = np.linalg.eigvals(self.state_matrix)
        spectral_radius = np.max(np.abs(eigenvalues))
        if spectral_radius >= 1:
            if scale_matrix:
                self.state_matrix /= (spectral_radius + 0.1)
            else:
                warnings.warn('Warning: state_matrix is not stable.')

    def copy_with_changes(self, **kwargs) -> Self:
        """
        Copy the initial parameters of this object into another object. Allow
        kwargs to change the initial values of the copy object.
        """

        # Copy current object
        other = copy.deepcopy(self)

        # Copy changed arguments
        for k in kwargs:
            other.__dict__[k] = kwargs[k]

        scale_key = "scale_matrix"
        if scale_key in kwargs:
            other.rescale_matrix(kwargs[scale_key])

        return other

    def evolve_state(self, state: NDArray) -> NDArray:
        """
        Docstring TODO
        """
        mv_n = self.rng.multivariate_normal
        draws = mv_n(np.zeros((self.d)), self.state_noise_cov,
                     size=(state.shape[0]))[:, :, np.newaxis]

        return self.state_matrix @ state + draws

    def get_obs(self, state: NDArray) -> NDArray:
        """
        Docstring TODO
        """
        mv_n = self.rng.multivariate_normal
        draws = mv_n(np.zeros((self.p)), self.obs_noise_cov,
                     size=(state.shape[0]))[:, :, np.newaxis]

        return self.obs_matrix @ state + draws

    def sample(self, n: int, t: int, init_mean: NDArray | None = None,
               init_cov: NDArray | None = None) -> NDArray:
        """
        Samples from LinearSSM with initial state sampled from Gaussian with
        init_mean and init_cov.
        Default: mean zero with unit variance.

        Parameters:
            n: number of time series replicates to sample
            t: number of time points in each time series
            init_mean: shape (d,1) mean values for initial state sampling
            init_cov: shape (d,1) covariance for initial state sampling

        Returns:
            Numpy array of shape (n,p,t) representing generated time series
        """
        self.ts_samples = np.zeros((n, self.p, t))
        if init_mean is None:
            init_mean = np.zeros((self.d))
        if init_cov is None:
            init_cov = np.eye(self.d)

        # Sample initial state
        state = self.rng.multivariate_normal(init_mean, init_cov,
                                             size=n)[:, :, np.newaxis]

        # Recursively sample observations
        for t_index in range(t):
            self.ts_samples[:, :, t_index] = self.get_obs(state)[:, :, 0]
            state = self.evolve_state(state)

        return self.ts_samples

    def plot_sample(self) -> None:
        """
        Plot the time series sample
        """
        plt.figure()

        for i in range(self.p):
            subplot_index = int(f"{self.p}1{i + 1}")
            plt.subplot(subplot_index)

            plt.plot(self.ts_samples[:, i, :].T)
            plt.title(f"Dim. {i}")

        plt.xlabel('Time')
        plt.show()


class HSMM(TimeSeries):

    def __init__(self, init_probs: NDArray, transition_probs: NDArray,
                 emission_means: list[NDArray],
                 emission_covariances: list[NDArray],
                 state_durations_params: list[tuple],
                 rng: np.random.Generator | None = None) -> None:
        """
        Initialize the Hidden Semi-Markov Model with multivariate emissions.

        Parameters:
            init_probs: A 1D numpy array of initial state probabilities
                (must sum to 1).
            transition_probs: A 2D numpy array (N x N) where N is the number of
                states. Each entry represents the probability of transitioning
                from one state to another.
            emission_means: A list of mean vectors (1D numpy arrays) for
                emissions in each state.
            emission_covariances: A list of covariance matrices
                (2D numpy arrays) for emissions in each state.
            state_durations_params: A list of tuples
                [(mean, std, min, max), ...] representing the mean, standard
                deviation, and truncation range for the duration of each state.
            rng: numpy rng (else use default); see
                https://numpy.org/doc/2.0/reference/random/index.html#random-quick-start
        """
        super().__init__()

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.init_probs = np.array(init_probs)
        self.transition_probs = np.array(transition_probs)
        self.emission_means = [np.array(mean) for mean in emission_means]
        self.emission_covariances = [np.array(cov) for cov in
                                     emission_covariances]
        self.state_durations_params = state_durations_params

        # Validate inputs
        num_states = len(self.init_probs)
        if not np.isclose(np.sum(self.init_probs), 1):
            raise ValueError("Initial state probabilities must sum to 1.")
        if self.transition_probs.shape[0] != self.transition_probs.shape[1]:
            raise ValueError("Transition matrix must be square.")
        if self.transition_probs.shape[0] != num_states:
            e = "Transition matrix dimensions must match the number of states."
            raise ValueError(e)
        if not np.allclose(self.transition_probs.sum(axis=1), 1):
            e = "Each row of the transition matrix must sum to 1."
            raise ValueError(e)
        if len(self.emission_means) != num_states:
            e = "Number of emission means must match the number of states."
            raise ValueError(e)
        if len(self.emission_covariances) != num_states:
            e = "Number of emission covariances must match the "
            e += "number of states."
            raise ValueError(e)
        if len(self.state_durations_params) != num_states:
            e = "Number of state duration parameters must match the "
            e += "number of states."
            raise ValueError(e)

        for cov in self.emission_covariances:
            if cov.shape[0] != cov.shape[1]:
                raise ValueError("Each covariance matrix must be square.")
            if not np.allclose(cov, cov.T):
                raise ValueError("Each covariance matrix must be symmetric.")
            if not np.all(np.linalg.eigvals(cov) >= 0):
                e = "Each covariance matrix must be positive semi-definite."
                raise ValueError(e)

    def truncated_discrete_normal(self, mean: float, std: float,
                                  min_val: float, max_val: float,
                                  size: int = 1) -> NDArray:
        """
        Sample from a truncated discrete normal distribution.

        Parameters:
            mean: Mean of the normal distribution.
            std: Standard deviation of the normal distribution.
            min_val: Minimum truncation value.
            max_val: Maximum truncation value.
            size: Number of samples to generate.

        Returns:
            A NumPy array of samples.
        """
        a = (min_val - mean) / std
        b = (max_val - mean) / std
        samples = truncnorm(a, b, loc=mean, scale=std).rvs(
            size=size, random_state=self.rng)

        return np.clip(np.round(samples), min_val, max_val).astype(int)

    def sample(self, n: int, t: int) -> tuple[list[NDArray], list[NDArray]]:
        """
        Generate n time series of length t from the Hidden Semi-Markov Model
        with multivariate emissions.

        Parameters:
            n: Number of time series to generate.
            t: Length of each time series.

        Returns:
            samples: A list of n numpy arrays, each of shape (t, D),
                representing the multivariate time series.
            states: A list of n numpy arrays, each of shape (t,), representing
                the hidden state sequence.
        """

        num_states = len(self.emission_means)
        samples = []
        states = []

        for _ in range(n):
            sequence: list[NDArray] = []
            state_sequence = []

            # Start in a random initial state
            current_state = self.rng.choice(num_states, p=self.init_probs)
            time = 0

            while time < t:
                # Sample duration from truncated discrete normal distribution
                values = self.state_durations_params[current_state]
                mean, std, min_val, max_val = values
                duration = self.truncated_discrete_normal(mean, std, min_val,
                                                          max_val, size=1)[0]

                # Limit duration to avoid exceeding the time series length
                duration = min(duration, t - time)

                # Generate multivariate emissions for the duration
                em = self.emission_means[current_state]
                ec = self.emission_covariances[current_state]
                emissions = self.rng.multivariate_normal(em, ec, duration)
                sequence.extend(emissions)
                state_sequence.extend([current_state] * duration)

                # Transition to the next state
                tp = self.transition_probs[current_state]
                next_state = self.rng.choice(num_states, p=tp)
                current_state = next_state

                time += duration

            # Append the generated sequence and states
            samples.append(np.array(sequence))
            states.append(np.array(state_sequence))

        return samples, states
