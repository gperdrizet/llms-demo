'''Chat bot demo using a local llama.cpp server or any other OpenAI-compatible API.


--- Option 1: Connect to an OpenAI-compatible remote server ---

1. Create a .env file in the repo root with the server address and your API key:

   OPENAI_API_URL=<server-address>
   OPENAI_API_KEY=<api-key>

2. Run the chatbot - it will automatically connect to the remote server:

   $ python demos/chatbots/llamacpp_chatbot.py


--- Option 2: Build and run the server locally ---

1. See the repo documentation for 
instructions on how to build llama.cpp and start the server.

2. Once the server is running, run the chatbot (no .env needed, defaults to localhost:8502 with API key "dummy"):

   $ python demos/chatbots/llamacpp_chatbot.py
'''

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration
temperature = 0.7

llamacpp_server  = os.environ.get('OPENAI_API_URL', 'localhost:8502')
llamacpp_api_key = os.environ.get('OPENAI_API_KEY', 'dummy')
llamacpp_model   = os.environ.get('OPENAI_API_MODEL', 'default')

system_prompt = (
    'Reasoning: low\n\n'
    'You are a helpful teaching assistant at an AI/ML boot camp. '
    'Answer questions in simple language with examples when possible.'
)

# Initialize the OpenAI client pointing at the llama.cpp server
client = OpenAI(
    base_url=llamacpp_server,
    api_key=llamacpp_api_key,
)

# Start conversation history with system prompt
history = [{'role': 'system', 'content': system_prompt}]

def main():
    '''Main conversation loop.'''

    print(f'Connected to inference server at {llamacpp_server}')
    print(f'Model: {llamacpp_model}')
    print('Type "exit" to quit.\n')

    # Loop until user types 'exit'
    while True:

        # Get text input from the user
        user_input = input('User: ')

        # Check for exit condition
        if user_input.lower() in ['exit', 'quit']:
            print('Exiting chatbot.')
            break

        # Add the user's message to the conversation history
        history.append({'role': 'user', 'content': user_input})

        # Stream the response so tokens appear as they are generated
        stream = client.chat.completions.create(
            model=llamacpp_model,
            messages=history,
            temperature=temperature,
            stream=True,
        )

        print(f'\n{llamacpp_model}: ', end='', flush=True)

        assistant_message = ''

        for chunk in stream:

            try:
                token = chunk.choices[0].delta.content

            except IndexError:
                token = None

            if token:
                print(token, end='', flush=True)
                assistant_message += token

        print('\n')

        # Add the model's response to the conversation history
        history.append({'role': 'assistant', 'content': assistant_message})


# Main entry point
if __name__ == '__main__':
    main()
