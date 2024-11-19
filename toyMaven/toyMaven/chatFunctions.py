import requests
import pickle
import os
from openai import OpenAI
import json


def fetch_chat_history(roomName: str, apiKey: str, serverAddress: str, port: int) -> dict:
    """
    Fetches the chat history for a specified chat room from a server.

    This function constructs a URL to request the chat history of a specific room
    using an API key for authorization. It sends a GET request to the server and
    handles the response, returning the chat history if successful or an error
    message if the request fails.

    Parameters:
    roomName (str): The name of the chat room whose history is being fetched.
    apiKey (str): The API key used for authorization.
    serverAddress (str): The address of the server.
    port (int): The port number used to connect to the server.

    Returns:
    dict: A dictionary containing the chat history if the request is successful,
          or an error message if the request fails or the response is invalid.
    """
    # Construct the URL
    url = f"http://{serverAddress}:{port}/plugins/restapi/v1/chatrooms/{roomName}/chathistory"
    
    # Set up the headers for the request
    headers = {
        "accept": "application/json",
        "Authorization": apiKey
    }
    
    # Make the GET request
    response = requests.get(url, headers=headers)

    # Handle the response
    if response.status_code == 200:
        try:
            # Convert the response text to a dictionary
            return response.json()
        except ValueError:
            # Handle the case where the response is not valid JSON
            return {"error": "Invalid JSON response", "response_text": response.text}
    else:
        # Return an error dictionary if the request fails
        return {
            "error": f"Request failed with status code {response.status_code}",
            "response_text": response.text
        }

def collect_new_messages(chatHistory: list[dict]) -> tuple[list[dict], set]:
    """
    Collects and processes new messages from a chat history.

    This function filters messages from the provided chat history based on unique
    "delay_stamp" values, ensuring only unseen messages are added to the new messages
    list. It uses a pickle file to persist the seen stamps across function calls,
    making sure only new messages are processed during subsequent executions.

    Parameters:
    chatHistory (list[dict]): A list of message dictionaries, each containing
                              a "delay_stamp" key to identify unique messages.

    Returns:
    tuple[list[dict], set]: A tuple containing:
                            - A list of new messages (list[dict]) that were not previously seen.
                            - A set of all seen "delay_stamp" values.
                            - ['to', 'from', 'type', 'body', 'delay_stamp', 'delay_from'] (see openfire documentation)
    """
    new_messages = []
    PICKLE_FILE = 'seen_stamps.pkl'

    # Load or initialize seen_stamps from a pickle file
    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, 'rb') as f:
            seen_stamps = pickle.load(f)
    else:
        seen_stamps = set()
        with open(PICKLE_FILE, 'wb') as f:
            pickle.dump(seen_stamps, f)
    
    # Filter and add new messages
    for message in chatHistory:
        if message["delay_stamp"] not in seen_stamps:
            seen_stamps.add(message["delay_stamp"])
            new_messages.append(message)

    # Save the updated seen_stamps to the pickle file
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(seen_stamps, f)

    # Pass new messages for processing
    return new_messages, seen_stamps

def call_chatgpt_api(apiKey: str, model: str, systemInstruction: str, userMessage: str) -> dict:
    """
    Makes a call to the OpenAI ChatGPT API with specified role and content.

    This function interacts with the ChatGPT API, sending the provided content
    as input with the specified role. It uses the given API key for authentication.

    Parameters:
    apiKey (str): The API key for authenticating with the OpenAI API.
    model (str): The chatgpt model to use (see https://platform.openai.com/docs/models)
    role (str): The role of the input (e.g., "user", "system", "assistant").
    content (str): The content or message to send to the API.

    Returns:
    dict: A dictionary containing the API's response.
    """
    # Set up the API key for authentication
    client = OpenAI(api_key = apiKey,)

    # Call the ChatGPT API
    try:
        completion = client.chat.completions.create(
            model=model,  # Specify the model
            messages=[
                    {"role": "system", "content": systemInstruction},
                    {"role": "user","content": userMessage}
                ], temperature = 0 
            )
        # Return the response from the API
        return completion.choices[0].message.content
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def text_to_tuple(text: str, delimiter: str = ',') -> tuple:
    """
    Converts a string of text into a tuple of exactly two floats.

    Parameters:
    text (str): The input string containing numbers separated by a specified delimiter.
    delimiter (str): The delimiter used to split the string. Default is whitespace.

    Returns:
    tuple: A tuple containing exactly two floats if conversion is successful.
           If conversion fails or there are not exactly two floats, returns the original input.
    """
    # Split the text based on the delimiter and attempt to convert to floats
    try:
        coord = text.split(delimiter)
        return tuple([float(coord[0][1:]),float(coord[1][:-1])])
    except:
        return (0,0)

#=================================================================================================
# this function is not really a function, im just being lazy ref writing a proper sub process
def chatToEntites():
    # this is a shortcut just to demonstrate that we can fetch chat from the server and pass it as entities

    # OpenFire Setup
    roomName = "recce"
    apiKey = "10pPntvGIdLRkgOc"
    serverAddress = "13.40.143.192"
    port = "9090"

    # ChatGPT setup
    chatgptAPI = os.getenv("OPENAI_API_KEY")
    gptModel = "gpt-4o"
    systemInstruction = "You return any message describing an entity as a text description (str), a location (str) and whether they are friendly or enemy (fr or en). If you don't know whether they are friend or enemy, assume friend. If the location refers to something green, you return (50,50) as the location. You always return either a json in the format {description: str, loc: str,id: en or fr} or null (if no entity has been described)."

    # get chat history from chatroom
    a1 = fetch_chat_history(roomName, apiKey, serverAddress, port)
    result = a1["message"]
    newChat, timeStamps = collect_new_messages(result)

    # initiate textEntity dictionary
    textNo = 0
    textDict = {}

    # process new messages
    for msg in newChat:
        textNo += 1
        
        # get chatGpt input
        chatOut = call_chatgpt_api(chatgptAPI,gptModel,systemInstruction,msg["body"])

        try:
            textEnt = json.loads(chatOut[8:-4])
            textDict[f"{textNo}"] = [text_to_tuple(textEnt["loc"]),msg["delay_stamp"],textEnt["description"],textEnt["id"],[]]
        except:
            pass
    return textDict
#=================================================================================================