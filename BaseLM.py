import os
import google.generativeai as genai 
import json
import yaml
import time
import google.api_core.exceptions


class BaseLM:
    """
    A base class for language models from a configuration file.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str,
        endpoint: str = None,
        api_version: str = None,
        max_completion_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str = "",
        **kwargs
    ):
        """
        Initialize the language model.

        :param provider: Provider name (e.g. "openai", "anthropic", "together").
        :param model_name: model name
        :param api_key: API key
        :param endpoint: (optional) provider endpoint
        :param api_version: (optional) API version
        :param max_completion_tokens: max completion tokens
        :param temperature: sampling temperature
        :param kwargs: any additional keyword arguments
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.extra_params = kwargs
        self.client = None
        self.system_prompt = system_prompt
        
        if self.provider == "gemini": self._init_gemini()
        else: raise ValueError(f"Unsupported provider: {self.provider}")

    
    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def _init_gemini(self):
        """Initialize Google Gemini API client."""
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model_name)

    
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text based on the given prompt using the configured provider and model.

        :param prompt: The text prompt for the model.
        :return: The generated text as a string.
        """
        if self.provider == "gemini":
            return self._generate_gemini(prompt)
        else: raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_gemini(self, prompt: str) -> str:
        """Generate text using Google Gemini API with unlimited retries until success."""
        attempt = 0  # Track number of attempts

        while True:
            try:
                response = self.client.generate_content(prompt)
                # Ensure a valid response exists
                if response and response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text.strip()

            except google.api_core.exceptions.ResourceExhausted:
                wait_time = min(2 ** attempt, 60)  # Exponential backoff, capped at 60 seconds
                print(f"⚠️ API quota exceeded, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1  # Increase wait time for next attempt

            except google.api_core.exceptions.GoogleAPIError as e:
                return "Error: GoogleAPI Error."
        
    @classmethod
    def from_config(cls, config_path: str):
        """
        Instantiate from a JSON config file.

        :param config_path: path toconfig file.
        :return: instance of BaseLM
        """
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path.endswith(".yaml"):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

        return cls(**config)

