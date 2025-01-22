import os
import json
from llama_apis.llama_apis import LLaMApis

def main():
    # Load API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Initialize the LLaMApis instance
    llm = LLaMApis(openai_api_key=openai_api_key, perplexity_api_key=None)

    # Define a simple prompt
    prompt = """Give me a short summary of Oxford in JSON format with fields: {"founding_year", "population", "famous_for"}"""

    # Call OpenAI API with JSON output
    response = llm.call_openai_api(prompt=prompt, as_json=True)

    # Print the JSON response
    print("\nGenerated JSON Output:")
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()
