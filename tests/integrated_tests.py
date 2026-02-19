import unittest
from pathlib import Path

import numpy as np
import ruptures as rpt
from numpy.typing import NDArray

from lambast.detection_methods import ChangePoint
from lambast.generate_data import (HSMM, ClaytonCopula, FrankCopula, JoeCopula,
                                   LinearSSM, NormalCopula, Voigt)


class IntegratedTests(unittest.TestCase):
    '''
    Generic class for integrated tests
    '''

    def __init__(self, *args):
        super().__init__(*args)

    def check_with_blessed(self, file: str | Path, array:  NDArray,
                           n_digits: int = 14, save: bool = False,
                           directory: str = "blessed_files") -> None:
        '''
        Generic function to test array against blessed file
        '''

        pathDir = Path(directory)

        # If not inside of the tests directory already, add it to the path
        dirname = Path.cwd().name
        if dirname != "tests":
            pathDir = "tests" / pathDir

        pathFile = pathDir / file

        if save:
            np.save(pathFile, array)

        blessed = np.load(f"{pathFile}.npy")
        diff = np.mean(np.abs(blessed - array))
        self.assertAlmostEqual(diff, 0, n_digits)


class LinearSSMTests(IntegratedTests):
    '''
    Class for testing linear_SSM
    '''

    def __init__(self, *args):
        super().__init__(*args)

        # Settings
        rng = np.random.default_rng(seed=42)
        self.d = 10
        self.p = 2
        self.n = 10
        self.t = 150

        # Generate "training data"
        state_matrix = rng.normal(size=(self.d, self.d))
        obs_matrix = rng.normal(size=(self.p, self.d))
        state_noise_cov = 0.1 * np.eye(self.d)
        obs_noise_cov = 0.01 * np.eye(self.p)

        # Instantiate the SSM object
        self.ssm = LinearSSM(state_matrix, state_noise_cov, obs_matrix,
                             obs_noise_cov, rng=rng, scale_matrix=True)
        # Sample the time series
        self.ssm.sample(self.n, self.t)

    def test_linear_SSM(self) -> None:
        '''
        Base case
        '''

        # Check against blessed file
        ssm_dir = Path("linear_SSM")
        self.check_with_blessed(ssm_dir / "base", self.ssm.ts_samples)

    def test_change_state_matrix(self) -> None:
        '''
        Changing the state matrix
        '''

        # Re-initialize the rng so this does not depend on
        # the order this is run
        rng = np.random.default_rng(seed=42)

        state_matrix = self.ssm.state_matrix + \
            0.1 * rng.normal(size=(self.d, self.d))
        ssm = self.ssm.copy_with_changes(state_matrix=state_matrix,
                                         scale_matrix=True, rng=rng)
        ssm.sample(self.n, self.t)

        # Check against blessed file
        ssm_dir = Path("linear_SSM")
        self.check_with_blessed(ssm_dir / "state_matrix", ssm.ts_samples)

    def test_change_observation_noise(self) -> None:
        '''
        Changing the observation noise
        '''

        # Re-initialize the rng so this does not depend on
        # the order this is run

        obs_noise_cov = 1.0 * np.eye(self.p)
        ssm = self.ssm.copy_with_changes(
            obs_noise_cov=obs_noise_cov, rng=np.random.default_rng(seed=42))
        ssm.sample(self.n, self.t)

        # Check against blessed file
        ssm_dir = Path("linear_SSM")
        self.check_with_blessed(ssm_dir / "observation_noise", ssm.ts_samples)

    def test_change_observation_matrix(self) -> None:
        '''
        Changing the observation matrix
        '''

        # Re-initialize the rng so this does not depend on
        # the order this is run
        rng = np.random.default_rng(seed=42)

        obs_matrix = self.ssm.obs_matrix + 1.5 * \
            rng.normal(size=(self.p, self.d))
        ssm = self.ssm.copy_with_changes(obs_matrix=obs_matrix, rng=rng)
        ssm.sample(self.n, self.t)

        # Check against blessed file
        ssm_dir = Path("linear_SSM")
        self.check_with_blessed(ssm_dir / "observation_matrix", ssm.ts_samples)


class CopulaTests(IntegratedTests):
    '''
    Class for testing ClaytonCopula
    '''

    def __init__(self, *args):
        super().__init__(*args)

        # Settings
        self.n = 4  # number of time-series to generate
        self.t = 400  # length of time series (arbitrary units)
        self.n_samples = 100  # Samples for the density

        # Generate copula sublcass paramter and markovian time series property
        # Dictionary of copulas
        self.copulas = {
            "Clayton": ClaytonCopula(alpha=10, markovian=True),
            "Joe": JoeCopula(alpha=10, markovian=True),
            "Frank": FrankCopula(alpha=10, markovian=True),
            "Normal": NormalCopula(alpha=0.5, markovian=True),
        }

        # Define the marginal distribution for the time series
        for c in self.copulas.values():
            c.define_marginal(marginal_family="gamma", loc=None, scale=None)

    def __test_samples(self, name: str) -> None:
        '''
        Test the named copula sample generation
        '''

        # Generate the samples with seed for test reproducibility
        rng = np.random.default_rng(seed=42)

        file = Path("copulas") / f"{name.lower()}_samples"
        copula = self.copulas[name.capitalize()]
        self.check_with_blessed(file, copula.sample(self.n, self.t, rng=rng))

    def __test_density(self, name: str) -> None:
        '''
        Test the named copula density generation
        '''

        file = Path("copulas") / f"{name.lower()}_density"
        copula = self.copulas[name.capitalize()]
        self.check_with_blessed(file, copula.density(n_samples=self.n_samples))

    def test_clayton_copula_samples(self) -> None:
        '''
        Test the Clayton copula sample generation
        '''

        self.__test_samples("Clayton")

    def test_clayton_copula_density(self) -> None:
        '''
        Test the Clayton copula density generation
        '''

        self.__test_density("Clayton")

    def test_joe_copula(self) -> None:
        '''
        Test the Joe copula sample generation
        '''

        self.__test_samples("Joe")

    def test_joe_copula_density(self) -> None:
        '''
        Test the Joe copula density generation
        '''

        self.__test_density("Joe")

    def test_frank_copula(self) -> None:
        '''
        Test the Frank copula sample generation
        '''

        self.__test_samples("Frank")

    def test_frank_copula_density(self) -> None:
        '''
        Test the Frank copula density generation
        '''

        self.__test_density("Frank")

    def test_normal_copula(self) -> None:
        '''
        Test the normal copula sample generation
        '''

        self.__test_density("normal")

    def test_normal_copula_density(self) -> None:
        '''
        Test the normal copula density generation
        '''

        self.__test_samples("normal")


class HSMMTests(IntegratedTests):
    '''
    Class for testing HSMM
    '''

    def __init__(self, *args):
        super().__init__(*args)

        # Settings
        # Define number of time series to generate and their length
        self.n = 5  # 5 time-series
        self.t = 100  # Length 100

        # Define HSMM parameters

        # Initial state probabilities
        init_probs = [0.7, 0.3]

        # Transition probabilities
        transition_probs = [[0.8, 0.2], [0.3, 0.7]]

        # Mean vectors for states 0 and 1
        emission_means = [[0, 0], [3, 3]]

        # Covariance matrices for state 0 and 1
        emission_covariances = [[[1, 0.2], [0.2, 1]],
                                [[1, -0.3], [-0.3, 1]]]

        # Duration parameters for each state
        state_durations_params = [
            (10, 3, 1, self.t),  # State 1: mean=10, std=3, min=5, max=t
            (20, 5, 1, self.t),  # State 2: mean=20, std=5, min=10, max=t
        ]

        # Initialize the HSMM
        rng = np.random.default_rng(seed=42)
        self.hsmm = HSMM(init_probs, transition_probs, emission_means,
                         emission_covariances, state_durations_params, rng)

    def test_HSMM_sample(self) -> None:
        '''
        Simple sampling test
        '''

        samples, states = self.hsmm.sample(self.n, self.t)

        # Check against blessed file
        hsmm_dir = Path("HSMM")
        self.check_with_blessed(hsmm_dir / "HSMM_sample", samples)
        self.check_with_blessed(hsmm_dir / "HSMM_states", states)


class VoigtTests(IntegratedTests):
    '''
    Class for testing VoigtSignal
    '''

    def __init__(self, *args):
        super().__init__(*args)

        # Settings
        self.rng = np.random.default_rng(seed=42)

        self.fs = 1./1.8e-05
        self.nt = 2048
        self.in_dist_range = [-200, 200]
        self.out_dist_range = [300, 700]
        self.sample_n = 100

        self.in_keys = ["nt", "fs", "freq", "phi", "decay_rate", "amp",
                        "sigma", "noise_var", "const", "t", "f_vec"]
        self.out_keys = ["snr", "sigs", "noise", "noisy_sig", "freqs"]

    def test_Voigt_sample_in(self) -> None:
        '''
        Simple synthetic data generation
        '''

        df_in, df_out = Voigt(sample_n=self.sample_n, nt=self.nt,
                              fs=1./1.8e-05, freq_range=self.in_dist_range,
                              phi_range=(-np.pi, np.pi),
                              decay_rate_range=(1e-3, 1e-2),
                              sigma_range=(1e-3, 1e-2), amp_range=(1, 2),
                              noise_var=1, rng=self.rng).synthetic_data_gen()

        voigt_dir = Path.joinpath(Path("Voigt"), "fr_in")
        for key in self.in_keys:
            self.check_with_blessed(voigt_dir / f"Voigt_in_{key}", df_in[key])

        for key in self.out_keys:
            self.check_with_blessed(
                voigt_dir / f"Voigt_out_{key}", df_out[key])

    def test_Voigt_sample_out(self) -> None:
        '''
        Simple synthetic data generation
        '''

        df_in, df_out = Voigt(sample_n=self.sample_n, nt=self.nt,
                              fs=1./1.8e-05, freq_range=self.out_dist_range,
                              phi_range=(-np.pi, np.pi),
                              decay_rate_range=(1e-3, 1e-2),
                              sigma_range=(1e-3, 1e-2), amp_range=(1, 2),
                              noise_var=1, rng=self.rng).synthetic_data_gen()

        voigt_dir = Path.joinpath(Path("Voigt"), "fr_out")
        for key in self.in_keys:
            self.check_with_blessed(voigt_dir / f"Voigt_in_{key}", df_in[key])

        for key in self.out_keys:
            self.check_with_blessed(
                voigt_dir / f"Voigt_out_{key}", df_out[key])


class ChangePointTest(IntegratedTests):
    '''
    Class for testing ChangePoint
    '''

    def _synthetic_data(self, gen_type: str = "constant", n_samples: int = 200,
                        n_features: int = 1, n_bkps: int = 3,
                        noise_std: float | None = None,
                        delta: tuple[int, int] = (1, 10),
                        seed: int | None = None) -> NDArray[np.float64]:
        """
        Create synthetic data to test changepoint detection, only the arguments
        compatible with the call will be preserved. The possible arguments are:

        gen_type: one of 'constant', 'linear', 'normal', 'wavy'
        n_samples: signal length
        n_features: number of dimensions
        n_bkps: number of changepoints
        noise_std: noise std. If None, no noise is added
        delta: (delta_min, delta_max) max and min jump values
        seed: random seed
        """

        generators = {
            "constant": [rpt.pw_constant, {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_bkps": n_bkps,
                "noise_std": noise_std,
                "delta": delta,
                "seed": seed,
            }],
            "linear": [rpt.pw_linear, {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_bkps": n_bkps,
                "noise_std": noise_std,
                "seed": seed,
            }],
            "normal": [rpt.pw_normal, {
                "n_samples": n_samples,
                "n_bkps": n_bkps,
                "seed": seed,
            }],
            "wavy": [rpt.pw_wavy, {
                "n_samples": n_samples,
                "n_bkps": n_bkps,
                "noise_std": noise_std,
                "seed": seed,
            }],
        }

        signal, bkps = generators[gen_type][0](**generators[gen_type][1])

        return signal

    def _detect_changepoints(self, signals: list, use_estimator: str,
                             kwarg_list: list) -> list:
        """
        Helper function to detect changepoints with a given estimate and plot
        them at the same time
        """

        # Initialize the change point detection object
        ts_detection = ChangePoint()

        # Detect and plot
        breakpoints = []
        for i, signal in enumerate(signals):
            ts_detection.detect_change_point(signal, estimator=use_estimator,
                                             **kwarg_list[i])

            breakpoints.append(ts_detection.optimal_breakpoints)

        return breakpoints

    def __init__(self, *args):
        super().__init__(*args)

        # Create the signals
        gen_type = "constant"
        n_features = 10
        n_bkps = 2
        seed = 42

        # Clean signals
        signal_clean = self._synthetic_data(
            gen_type=gen_type, n_bkps=n_bkps, seed=seed, n_features=n_features)

        # Add some noise
        noise_std = 1
        signal_noise_low = self._synthetic_data(
            gen_type=gen_type, n_bkps=n_bkps, noise_std=noise_std, seed=seed,
            n_features=n_features)

        # Add much more noise
        noise_std = 10
        signal_noise_high = self._synthetic_data(
            gen_type=gen_type, n_bkps=n_bkps, noise_std=noise_std, seed=seed,
            n_features=n_features)

        self.signals = [signal_clean, signal_noise_low, signal_noise_high]

    def test_pelt(self) -> None:
        """
        Test PELT
        """

        use_estimator = "Pelt"
        kwarg_list = [{"pen": 2.}, {"pen": 4.}, {"pen": 8.}]
        breakpoints = self._detect_changepoints(
            self.signals, use_estimator, kwarg_list)

        assert breakpoints[0] == [70, 135, 200]
        assert breakpoints[1] == [70, 135, 200]
        assert breakpoints[2] == [70, 200]

    def test_binseg(self) -> None:
        """
        Test Binary Segmentation
        """

        use_estimator = "Binseg"
        n_bkps = 2
        kwarg_list = [{"n_bkps": n_bkps},
                      {"n_bkps": n_bkps}, {"n_bkps": n_bkps}]
        breakpoints = self._detect_changepoints(
            self.signals, use_estimator, kwarg_list)

        assert breakpoints[0] == [70, 135, 200]
        assert breakpoints[1] == [70, 135, 200]
        assert breakpoints[2] == [70, 135, 200]


if __name__ == "__main__":
    unittest.main()
