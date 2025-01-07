"""

example usage
llm = LLaMApis(perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"), openai_api_key=os.getenv("OPENAI_API_KEY"))

out = llm.call_perplexity_api(prompt, as_json=True,
                               json_formatting=""" '{full_perplexity_text: str,  company_name: str, co2: float, no2:float, so2: float \} '""")

out = llm.call_openai_api(prompt, as_json=True)

                               
                           

"""

import json
import requests
from openai import OpenAI


class LLaMApis:
    def __init__(self, openai_api_key=None, perplexity_api_key=None):
        """
        Initialize the LLaMApis object.

        :param openai_api_key: Your OpenAI API key.
        :param perplexity_api_key: Your Perplexity API key.
        """
        print('initialising')
        self.openai_api_key = openai_api_key
        self.perplexity_api_key = perplexity_api_key

        # Initialize the OpenAI client if API key is provided
        self.openai_client = None
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)

        self.response = None
        self.response_openai = None


    def convert_to_json(
        self,
        llm_output_text: str,
        json_formatting: str = "",
        model: str = "gpt-3.5-turbo-1106",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        """
        Converts LLM output text to a JSON object using OpenAI API.

        :param llm_output_text: The output text from any LLM (e.g., Perplexity, Claude).
        :param json_formatting: A string that defines the desired JSON structure.
        :param model: The OpenAI model to use for conversion (default "gpt-3.5-turbo-1106").
        :param temperature: Sampling temperature for the conversion (default 0.0 for deterministic output).
        :param max_tokens: Maximum tokens for the conversion response (default 512).
        :return: A dictionary representing the JSON-formatted output.
        """
        if not self.openai_client:
            raise ValueError(
                "No OpenAI client available for JSON formatting. "
                "Provide openai_api_key if you want to use `as_json=True`."
            )

        # Create a generalized prompt for JSON conversion
        json_format_prompt = (
            f"Convert the following text to JSON format using the specified structure:\n\n"
            f"Text:\n{llm_output_text}\n\n"
            f"""JSON Structure:\n{json_formatting}"""
        )

        # Call the OpenAI API to perform the conversion
        reformatted = self.call_openai_api(
            prompt=json_format_prompt,
            model=model,
            as_json=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return reformatted
    

    def call_openai_api(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        max_tokens: int = 256,
        as_json: bool = True,
        json_formatting: str = None,
        **kwargs
    ):
        """
        Calls the OpenAI API and returns a response.

        :param prompt: The prompt you want to send to OpenAI.
        :param model: The model name (default "gpt-4o-mini"). 
                      Change to your actual default model.
        :param temperature: Sampling temperature (default 0.5).
        :param max_tokens: Max tokens for response generation (default 256).
        :param as_json: If True, parse the response as JSON. Otherwise, return text.
        :param kwargs: Any additional parameters to pass to the OpenAI API.
        :return: JSON object or text string, depending on `as_json`.
        """
        if not self.openai_client:
            raise ValueError("OpenAI client is not initialized. Provide openai_api_key.")

        # Determine the response format
        response_format = {"type": "json_object"} if as_json else {"type": "text"}

        # Prepare the messages for the chat completion
        messages = [
            {
                "role": "system",
                "content": "Answer in a concise and accurate way with factual details.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # Base parameters for the API call
        base_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }

        # Update base_params with any additional kwargs
        base_params.update(kwargs)

        # Make the API call
        self.response_openai = self.openai_client.chat.completions.create(**base_params)

        # Extract the content from the response
        output = self.response_openai.choices[0].message.content.strip()

        print(base_params)
        if base_params['logprobs'] == True:
            print('Yes logprobs')
            self.logprobs = [token.logprob for token in self.response_openai.choices[0].logprobs.content]
        # Parse JSON if required
        if as_json:
            # Remove code block formatting if present
            if output.startswith("```json") and output.endswith("```"):
                output = output[len("```json\n"):-len("```")].strip()

            json_output = json.loads(output)

                    # **Start of Added Lines**
            if (base_params['logprobs'] == True):
                json_logprobs = self.extract_logprobs(
                    json_output=json_output,
                    original_text=prompt,
                    logprobs_data=self.response_openai.choices[0].logprobs
                )
                return {
                    "json_output": json_output,
                    "logprobs": json_logprobs
                }

            return json_output
        else:
            return output


    def call_perplexity_api(
        self,
        prompt: str,
        model: str = "llama-3.1-sonar-small-128k-online",
        temperature: float = 0.2,
        top_p: float = 0.9,
        as_json: bool = False,
        json_formatting: str = None,
        **kwargs
    ):
        """
        Calls the Perplexity API and returns a response.

        :param prompt: The user prompt you want to send to Perplexity.
        :param model: Model name (default "llama-3.1-sonar-small-128k-online").
        :param temperature: Sampling temperature (default 0.2).
        :param top_p: Top-p for nucleus sampling (default 0.9).
        :param as_json: If True, reformat the output as JSON using OpenAI.
        :param json_formatting: A string defining the desired JSON structure.
        :param kwargs: Additional parameters for the Perplexity API.
        :return: The raw text from Perplexity or a JSON object if as_json=True.
        """
        if not self.perplexity_api_key:
            raise ValueError("Perplexity API key is not initialized. Provide perplexity_api_key.")

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json",
        }

        # Base payload with default parameters
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "return_citations": True,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "hour",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1,
        }

        # Update payload with any additional kwargs
        payload.update(kwargs)

        # Make the API request to Perplexity
        self.response = requests.post(url, json=payload, headers=headers)
        self.response.raise_for_status()

        # Extract the content from the response
        perplexity_output_text = self.response.json()["choices"][0]["message"]["content"].strip()

        # If JSON formatting is requested, convert the output using OpenAI
        if as_json:
            if not json_formatting:
                raise ValueError("JSON formatting string must be provided when `as_json=True`.")

            # Use the new convert_to_json method
            reformatted = self.convert_to_json(
                llm_output_text=perplexity_output_text,
                json_formatting=json_formatting
            )
            return reformatted

        return perplexity_output_text
    






    def extract_logprobs(
        self,
        json_output: dict,
        original_text: str,
        logprobs_data: dict
    ):
        """
        Extracts log probabilities for tokens that are part of the JSON values.

        :param json_output: The JSON object generated from the LLM output.
        :param original_text: The original text output from the LLM before JSON conversion.
        :param logprobs_data: The logprobs data from the OpenAI API response.
        :return: A dictionary mapping JSON values to their respective token logprobs.
        """
        import re

        if not logprobs_data:
            raise ValueError("No logprobs data provided.")

        # Flatten the JSON values into a single string
        values = []
        for key, value in json_output.items():
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries if needed
                nested_values = self._flatten_json_values(value)
                values.extend(nested_values)
            # Add more conditions if your JSON can contain lists or other types

        combined_values_text = " ".join(values)

        # Tokenize the combined values text
        # Note: The tokenization method should match OpenAI's tokenization
        # For simplicity, we'll use a basic split. For accurate tokenization, consider using OpenAI's tiktoken library.
        tokens = combined_values_text.split()

        # Extract logprobs from the logprobs_data
        token_logprobs = logprobs_data.get("token_logprobs", [])

        if len(tokens) != len(token_logprobs):
            print("Warning: Number of tokens and logprobs do not match.")
            min_length = min(len(tokens), len(token_logprobs))
            tokens = tokens[:min_length]
            token_logprobs = token_logprobs[:min_length]

        # Map tokens to their logprobs
        token_logprob_mapping = {token: logprob for token, logprob in zip(tokens, token_logprobs)}

        # Optionally, map logprobs back to JSON structure
        # Here, we'll create a dict where each key maps to a list of (token, logprob)
        json_logprobs = {}
        for key, value in json_output.items():
            if isinstance(value, str):
                value_tokens = value.split()
                json_logprobs[key] = [(token, token_logprob_mapping.get(token, None)) for token in value_tokens]
            elif isinstance(value, dict):
                # Handle nested dictionaries
                nested_logprobs = self._extract_nested_logprobs(value, token_logprob_mapping)
                json_logprobs[key] = nested_logprobs
            # Add more conditions if your JSON can contain lists or other types

        return json_logprobs

    def _flatten_json_values(self, nested_dict):
        """
        Recursively flattens nested JSON dictionaries to extract string values.

        :param nested_dict: A nested dictionary.
        :return: A list of string values.
        """
        values = []
        for key, value in nested_dict.items():
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, dict):
                values.extend(self._flatten_json_values(value))
            # Add more conditions if your JSON can contain lists or other types
        return values

    def _extract_nested_logprobs(self, nested_dict, token_logprob_mapping):
        """
        Recursively extracts logprobs for nested JSON dictionaries.

        :param nested_dict: A nested dictionary.
        :param token_logprob_mapping: A dict mapping tokens to logprobs.
        :return: A nested dictionary mapping keys to lists of (token, logprob).
        """
        nested_logprobs = {}
        for key, value in nested_dict.items():
            if isinstance(value, str):
                value_tokens = value.split()
                nested_logprobs[key] = [(token, token_logprob_mapping.get(token, None)) for token in value_tokens]
            elif isinstance(value, dict):
                nested_logprobs[key] = self._extract_nested_logprobs(value, token_logprob_mapping)
            # Add more conditions if your JSON can contain lists or other types
        return nested_logprobs