from abc import ABCMeta, abstractmethod

import numpy as np


class Normalizer(metaclass=ABCMeta):
    """
    Abstract base class for online data normalizers.

    This class defines the standard API for all normalizers in the onorm package.
    All normalizers support incremental (online) learning, where the normalization
    parameters are updated as new data arrives without storing historical data.

    All concrete normalizer implementations must implement the abstract methods:
    partial_fit, transform, and reset.

    Examples
    --------
    ```{python}
    from onorm import MinMaxScaler
    import numpy as np
    scaler = MinMaxScaler(n_dim=3)
    for x in np.random.normal(size=(100, 3)):
        scaler.partial_fit(x)
    x_new = np.array([1.0, 2.0, 3.0])
    x_normalized = scaler.transform(x_new)
    ```
    """

    def __init__(self, **kwargs: object) -> None:
        """Initialize the normalizer."""
        pass

    @abstractmethod
    def partial_fit(self, x: np.ndarray) -> None:
        """
        Incrementally update the normalization model with a new observation.

        This method updates the internal state of the normalizer based on a new
        observation without storing the observation itself. This enables online
        learning with bounded memory usage.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array representing a new observation. The length should match
            the n_dim parameter used during initialization.

        Notes
        -----
        This method modifies the normalizer's internal state but does not
        transform the input data. Use `transform` or `partial_fit_transform`
        to normalize data.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform data using the current normalization model.

        This method applies the normalization transformation based on the
        statistics learned from previous observations via `partial_fit`.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array representing an observation to normalize. The length
            should match the n_dim parameter used during initialization.

        Returns
        -------
        np.ndarray
            The normalized observation as a 1-D array with the same shape as input.

        Notes
        -----
        This method modifies the input array in-place for efficiency. If you need
        to preserve the original array, pass a copy: `transform(x.copy())`.
        """
        raise NotImplementedError

    def partial_fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Update the normalization model and transform the data in one step.

        This is a convenience method equivalent to calling `partial_fit(x)`
        followed by `transform(x)`. It updates the model with the new observation
        and returns the normalized version.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array representing a new observation to fit and normalize.

        Returns
        -------
        np.ndarray
            The normalized observation as a 1-D array with the same shape as input.

        Notes
        -----
        The normalization is based on the statistics AFTER incorporating the
        new observation. This means the first observation will typically not
        be well-normalized since the model has minimal data.
        """
        self.partial_fit(x)
        return self.transform(x)

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the normalizer to its initial state.

        This method clears all learned statistics and returns the normalizer
        to its initial state, as if no observations have been seen. After
        calling reset(), the normalizer can be used on a new dataset.

        Notes
        -----
        This is useful when you want to reuse the same normalizer object on
        a completely different dataset without creating a new instance.
        """
        raise NotImplementedError
