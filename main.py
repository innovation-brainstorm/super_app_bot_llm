import os
from openai import OpenAI
import json
from functions import send_email,get_current_weather

# Configure proxy settings
proxy = 'http://172.28.0.1:10811'
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy


class Bot(object):

    def __init__(self,api_key,function_def,function_map):
        self.client = OpenAI(api_key=api_key)
        self.function_def=function_def # definition of functions
        self.function_map=function_map # map of function name and corresponse function
        # create message list to record historical chats. init with system prompt for the model to ask it don't make values
        self.messages=[{"role": "system", "content": "Don’t make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."}]
    

    def run_conversation(self,message):

        # append user's new message
        self.messages.append({"role": "user", "content": message})

        # step1: send to openai
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=self.messages,
            tools=self.function_def,
            tool_choice="auto",  
        )

        response_message = response.choices[0].message

        # extend conversation with assistant's reply
        self.messages.append(response_message) 


        # Step 2: check if the model wanted to call a function
        tool_calls = response_message.tool_calls
        if tool_calls:
            # Step 3: call the function
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.function_map[function_name]
                function_args = json.loads(tool_call.function.arguments)

                print("function call:\n {function_name}\n {function_args}")

                function_response = function_to_call(**function_args)

                # extend conversation with function response
                self.messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  

            # get a new response from the model where it can see the function response
            second_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=self.messages,
            )  

            response_message = second_response.choices[0].message

            self.messages.append(response_message)

        return response_message
    

if __name__=="__main__":
    function_def = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "send email for user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "_from": {
                            "type": "string",
                            "description": "the sender's email address",
                        },
                        "to": {"type": "string","description":"recipient email address. If multiple recipient provided, use ; to concat"},
                        "body": {"type": "string","description":"email body"},
                        "subject": {"type": "string","description":"email subject"},
                    },
                    "required": ["to","_from","body","subject"],
                },
            },
        }
    ]

    function_map = {
        "get_current_weather": get_current_weather,
        "send_email":send_email
    } 

    with open(".key","r") as f:
        key=f.read()
    
    bot=Bot(key,function_def,function_map)

    while 1:
        user_query=input()
        print(bot.run_conversation(user_query))
