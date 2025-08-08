# --- Import Necessary Libraries ---
# The 'openai' library is the official Python client for interacting with the OpenAI API.
from openai import OpenAI
# The 'os' library is used to interact with the operating system, here specifically to access environment variables.
import os
# The 'dotenv' library is used to load environment variables from a .env file into the os.environ dictionary.
# This is a security best practice, as it prevents you from hard-coding sensitive information like API keys in your code.
from dotenv import load_dotenv
# The 'json' library is re-imported here. While not strictly necessary to import it inside the loop,
# it's harmless. In a final script, this import would be moved to the top.
import json

# --- 1. Configuration and API Key Loading ---
# Load variables from a .env file in the same directory.
# The .env file should contain a line like: OPENAI_API_KEY="sk-..."
load_dotenv()

# Retrieve the API key from the environment variables.
# The script will fail here if the key is not found, which is a good "fail-fast" behavior.
openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize the OpenAI client object, authenticating with the provided API key.
# All subsequent API calls will be made through this 'client' object.
client = OpenAI(api_key=openai_api_key)

# --- 2. Data Generation Loop ---
# This loop will run 100 times to generate 100 distinct insurance documents.
# In a real-world scenario, you might generate thousands or millions of synthetic documents
# to train a machine learning model or populate a test database.
for i in range(100):
    print(f"Generating document {i}...")  # Provides progress feedback to the console.

    # --- 3. OpenAI API Call ---
    # This is the core of the script where we ask the OpenAI model to generate a document.
    response = client.chat.completions.create(
        # --- Model Selection ---
        # "gpt-4o-mini" is OpenAI's fast, multimodal, and cost-effective model. It's an excellent choice for
        # structured data generation tasks like this one, offering a great balance of speed, cost, and quality.
        # Alternative: "gpt-4-turbo" for potentially higher-quality or more complex generations, but at a higher cost.
        model="gpt-4o-mini",

        # --- Message Structure (Prompt Engineering) ---
        # The 'messages' parameter defines the conversation history given to the model.
        # This is a "few-shot" prompt, where we provide one or more examples to guide the model's output.
        messages=[
            {
                # The 'system' role sets the overall behavior, context, and high-level instructions for the AI.
                # This prompt is highly detailed to ensure the generated content is realistic and varied.
                "role": "system",
                "content": "Generate realistic documents related to insurance contracts. The documents can include: quote requests, policy issuance, claim reports, renewal notices, payment reminders, policy change requests, and other communications between insurance companies, agents, and clients. Make up the details: the type of policy (e.g., auto, home, accident, life, professional liability), the policyholder's data, the circumstances of any claim, and the contractual conditions. Include credible financial details such as the premium amount, coverage limits, deductibles, and any settlement figures. The documents must be detailed, simulating real correspondence. You must format the document in JSON."

            },
            {
                # The 'user' role simulates a user's direct request in the conversation.
                "role": "user",
                "type": "text",
                "content": "generate a document"
            },
            {
                # The 'assistant' role provides an example of the exact output format and style we want.
                # This is a very powerful technique (few-shot learning) that dramatically improves the
                # reliability and quality of the model's generations, especially for structured data like JSON.
                "role": "assistant",
                "content": [
                {
                "type": "text",
                "text": "{\"date\":\"August 7, 2025, 09:15 CEST\",\"subject\":\"Claim Opened and Document Request - Policy No. 45B-2024-HOME\",\"from\":\"claims.settlements@secure-insurance.com\",\"body\":\"Dear Mr. Green,\\n\\nWe are writing in reference to your communication of August 5, 2025, regarding the water damage that occurred at the property located at 10 Rome Street, insured under the policy in question (No. 45B-2024-HOME).\\n\\nWe have formally opened a claim file with the reference number 2025-CLM-98765. As per the terms of your contract, your 'Home Protected Plus' policy includes a deductible of €250.00 for this type of damage.\\n\\nTo proceed with the assessment and subsequent settlement of the damage, please send us the following documentation within 15 days of this document:\\n\\n1. Detailed photos of the damage to the property.\\n2. A cost estimate for the necessary repairs, issued by a construction company or a qualified contractor.\\n3. Any invoices for emergency work already performed.\\n\\nOne of our adjusters will contact you in the coming days to schedule an on-site inspection. You can send the documentation in reply to this Document or upload it directly to your client portal on our website.\\n\\nWe remain at your complete disposal for any clarification.\\n\\nSincerely,\\n\\nMs. Elena Gallo\\nClaims Settlement Department\\nSecure Insurance Inc.\\nPhone: 800 123 456\"}"
                }
                ]
                }
                ],


        # --- Response Format Enforcement ---
        # This parameter forces the model to output a valid JSON object that conforms to a specific schema.
        # This is crucial for creating reliable structured data.
        response_format={
            "type": "json_schema",  # Specifies that we are using the JSON Schema enforcement mode.
            "json_schema": {
                "name": "contract_schema",
                # `strict: True` is a powerful feature that ensures the output strictly follows the schema.
                # The model will make a best effort to only return the defined properties and match their types.
                "strict": True,
                "schema": {
                    "type": "object",
                    # `properties` defines the expected keys in the JSON object and their expected data types.
                    "properties": {
                        "date": {"type": "string", "description": "The date and time when the document was sent."},
                        "subject": {"type": "string", "description": "The subject line of the document."},
                        "from": {"type": "string", "description": "The sender's document address."},
                        "body": {"type": "string", "description": "The main content or body of the document."}
                    },
                    # `required` specifies which keys MUST be present in the generated JSON.
                    "required": ["date", "subject", "from", "body"],
                    # `additionalProperties: False` prevents the model from adding any extra keys not defined in 'properties'.
                    "additionalProperties": False
                }
            }
        },

        # --- Generation Parameters (Controlling Creativity) ---
        # `temperature` (0-2): Controls randomness. Higher values like 1.0 make the output more diverse and creative.
        # Lower values (e.g., 0.2) make it more focused and deterministic. 1.0 is good for generating varied examples.
        temperature=1,
        # `max_completion_tokens`: The maximum number of tokens (parts of words) to generate in the response.
        max_completion_tokens=2048,
        # `top_p`: An alternative to temperature for controlling randomness. A value of 1.0 means it is effectively disabled.
        top_p=1,
        # `frequency_penalty` & `presence_penalty` (-2 to 2): Used to discourage token repetition.
        # 0 is the default and means no penalty is applied, which is fine for this use case.
        frequency_penalty=0,
        presence_penalty=0
    )

    # --- 4. Process and Save the Response ---


    # Extract the generated content from the response object.
    # The path is `response.choices[0].message.content`.
    json_string = response.choices[0].message.content

    # Parse the JSON string into a Python dictionary. This allows us to manipulate it as a standard Python object.
    contract_data = json.loads(json_string)

    # Open a file in write mode ('w') and save the generated data.
    # The file is named using an f-string to include the current loop index (e.g., contract_0.json, contract_1.json, etc.).
    # `with open(...)` is used as it automatically handles closing the file, even if errors occur.
    # `encoding='utf-8'` ensures that special characters (like accents or currency symbols) are saved correctly.
    # `json.dump` serializes the Python dictionary back into a JSON formatted string in the file.
    # `ensure_ascii=False` allows non-ASCII characters to be written as-is (e.g., '€' instead of '\u20ac').
    # `indent=4` makes the saved JSON file human-readable by adding indentation.
    with open(f'data/contract_{i}.json', 'w', encoding='utf-8') as f:
        json.dump(contract_data, f, ensure_ascii=False, indent=4)

print("\nData generation complete.")