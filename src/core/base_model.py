import abc
import inspect
import json
import logging
import os

import numpy as np
import requests

logger = logging.getLogger(__name__)

class BaseModel(abc.ABC):
    """
    Abstract base class for a Watermarking model.

    All watermarking models must implement the `embed` and `detect` methods.
    Each model must have a `config.json` file in its respective directory.
    """

    def __init__(self):
        """
        Initializes the watermarking model by loading its configuration file.

        - Determines the file path of the subclass implementing this base class.
        - Constructs the path to `config.json` in the model's directory.
        - Loads the configuration file, raising an error if it is missing.
        """
        model_file = inspect.getfile(self.__class__)
        model_dir = os.path.dirname(os.path.abspath(model_file))

        self.config_path = os.path.join(model_dir, "config.json")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config.json not found in {self.config_path}")

        with open(self.config_path, "r") as json_file:
            self._config = json.load(json_file)

        self.base_url = None # subclasses must set this
    
    def _make_request(self, endpoint: str, json_data: dict, method: str = "POST", timeout: int = 300) -> dict:
        """
        Helper method to make HTTP requests to the model's backend service.

        Args:
            endpoint (str): The specific API endpoint path (e.g., '/embed').
            json_data (dict): The JSON payload to send.
            method (str): The HTTP method (e.g., 'POST', 'GET'). Defaults to 'POST'.
            timeout (int): Request timeout in seconds. Defaults to 300.

        Returns:
            dict: The JSON response from the server.

        Raises:
            ValueError: If self.endpoint is not set by the subclass.
            requests.exceptions.RequestException: For connection errors, timeouts, etc.
            requests.exceptions.HTTPError: For bad HTTP responses (4xx, 5xx).
            KeyError: If the response is not valid JSON or missing expected keys (handled by caller).
        """
        if not self.base_url:
            raise ValueError(f"self.base_url must be set in the __init__ method of {self.__class__.__name__}")

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"


        try:
            #logger.info(f"Making {method} request to {url} with data: {json_data}")
            logger.info("Making {method} request to {url} with data: [REDACTED]")
            response = requests.request(method, url, json=json_data, timeout=timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            raise # Re-raise the exception for the caller to handle
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode JSON response from {url}: {e}")
             raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during request to {url}: {e}")
            raise

    @abc.abstractmethod
    def embed(
        self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        """
        Embeds a watermark into the given audio signal.

        Args:
            audio (np.ndarray): The input audio signal.
            watermark_data (np.ndarray): The binary watermark data to be embedded.
            sampling_rate (int): The sampling rate of the audio signal.

        Returns:
            np.ndarray: The watermarked audio signal.

        This method must be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Detects (extracts) the watermark from the given audio signal.

        Args:
            audio (np.ndarray): The input audio signal containing a possible watermark.
            sampling_rate (int): The sampling rate of the audio signal.

        Returns:
            np.ndarray: The extracted watermark bits.

        This method must be implemented by subclasses.
        """
        pass

    def generate_watermark(self) -> np.ndarray:
        """
        Generates a sample watermark.

        Returns:
            np.ndarray: A randomly generated binary watermark with a length 
                        specified in the model's configuration (`config.json`).
        """
        return np.random.randint(
            0, 2, size=self.config["watermark_size"], dtype=np.int32
        )

    @property
    def name(self) -> str:
        """
        Returns the name of the watermarking model.

        Returns:
            str: The class name of the model instance.
        """
        return self.__class__.__name__

    @property
    def config(self) -> dict:
        """
        Provides read-only access to the model's configuration.

        Returns:
            dict: The model's configuration loaded from `config.json`.
        """
        return self._config