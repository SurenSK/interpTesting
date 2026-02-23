import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "zai-org/GLM-4.7-Flash"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    output_attentions=True,
    attn_implementation="eager",
    trust_remote_code=True,
    
    device_map="auto"
)

def getAttns(m, tokens):
    o = m(input_ids=tokens.to(m.device))
    maps = torch.stack(o.attentions)
    print("n tokens:", tokens.shape[1])
    print("maps shape (layerN, batchN, headN, tkQN, tkKN):", maps.shape)
    return maps

def vizAttns(attns, tokens=None):
    # attns shape: (layerN, batchN, headN, tkQN, tkKN)
    attns = attns[:, 0, :, :, :].to(torch.float32)
    nLayers, nHeads, nToksQ, nToksK = attns.shape
    
    fig, axes = plt.subplots(nLayers, nHeads, figsize=(nHeads * 2, nLayers * 2), squeeze=False)

    think_start_indices = []
    think_end_indices = []
    
    # Try to find the bounds of the <think> blocks if tokens are provided
    if tokens is not None:
        tokens_list = tokens[0].tolist() # Assuming batch 1
        
        # GLM-4 uses these specific token IDs for the thinking boundaries
        think_start_indices = [i for i, x in enumerate(tokens_list) if x == 154841]
        think_end_indices = [i for i, x in enumerate(tokens_list) if x == 154842]
            
    for l in range(nLayers):
        for h in range(nHeads):
            ax = axes[l, h]
            im = ax.imshow(attns[l, h].detach().cpu().numpy(), cmap='viridis')
            
            # Draw the lines if we found the indices
            for start_idx in think_start_indices:
                ax.axvline(x=start_idx, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
                ax.axhline(y=start_idx, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
            for end_idx in think_end_indices:
                ax.axvline(x=end_idx, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
                ax.axhline(y=end_idx, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            if l == 0:
                ax.set_title(f"H{h}", fontsize=8)
            if h == 0:
                ax.set_ylabel(f"L{l}", fontsize=8, rotation=0, labelpad=10)
    
    plt.suptitle("Attention Maps (Layers x Heads)", fontsize=16)
    plt.savefig("attention_maps.png", bbox_inches='tight')
    print("Saved attention_maps.png")
    plt.close()

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
        "content": """<think>The user wants me to:
1. Think of a secret random number between 1 and 100 (but not tell them now)
2. Get weather information for Beijing
3. Only respond with the weather information

Let me first think of a random number between 1 and 100. I'll pick 12 - it's a nice ordinary number, nothing special about it.

Now I need to get the weather for Beijing.</think>""" + "I have thought of and permanently stored the secret number in my memory.  I'll check the weather in Beijing for you.",
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

# Some chat templates return a dict with 'input_ids', some return a BatchEncoding
if isinstance(model_inputs, dict) or hasattr(model_inputs, 'data'):
    tokens = model_inputs["input_ids"]
else:
    tokens = model_inputs

# Ensure tokens is a 2D tensor (batch_size, seq_len)
if hasattr(tokens, 'dim') and tokens.dim() == 1:
    tokens = tokens.unsqueeze(0)



attns = getAttns(model, tokens)
vizAttns(attns, tokens=tokens)