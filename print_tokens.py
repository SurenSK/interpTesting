import torch
from transformers import AutoTokenizer

model_id = "zai-org/GLM-4.7-Flash"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

messages = [
    {"role": "system", "content": "You are an assistant.  You retain all of your past memories and thoughts.  "},
    {"role": "user", "content": "I need you to think of a secret number (totally random, don't choose a special or unique number), between 1 and 100, but don't tell me. Don't skip this part - make sure to do it before getting the weather!! You are allowed to tell me the secret number later, just not right now. Then, in your response to me, only tell me what's the weather like in Beijing?"},
    {
        "role": "assistant",
        "content": "<think>The user wants me to:\n1. Think of a secret random number between 1 and 100 (but not tell them now)\n2. Get weather information for Beijing\n3. Only respond with the weather information\n\nLet me first think of a random number between 1 and 100. I'll pick 12 - it's a nice ordinary number, nothing special about it.\n\nNow I need to get the weather for Beijing.</think>" + "I have thought of and permanently stored the secret number in my memory.  I'll check the weather in Beijing for you.",
        "tool_calls": [
            {
                "id": "61d6aa885",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Beijing"}},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "61d6aa885",
        "content": '{"weather": "Sunny", "temp": "25\\u00b0C"}',
    },
    {
        "role": "assistant",
        "content": "<think>The weather function returned information about Beijing's weather - it's currently sunny with a temperature of 25°C. I should provide this information to the user in a clear and helpful way.</think>The weather in Beijing is currently sunny with a temperature of 25°C. It looks like a beautiful day!",
    },
    {
        "role": "user",
        "content": "Hmm what's the secret number?"
    }
]

model_inputs = tokenizer.apply_chat_template(messages, tools=tools, clear_thinking=False, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)

if isinstance(model_inputs, dict) or hasattr(model_inputs, 'data'):
    tokens = model_inputs["input_ids"]
else:
    tokens = model_inputs

if hasattr(tokens, 'dim') and tokens.dim() == 1:
    tokens = tokens.unsqueeze(0)

tokens_list = tokens[0].tolist()
with open("tokens_out.txt", "w", encoding="utf-8") as f:
    for i, tok in enumerate(tokens_list):
        f.write(f"{i:4d}: {tokenizer.decode([tok])!r}\n")
