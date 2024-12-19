

def abstractify_prompt_fn(conversation, input):
    return f"""You will be given a conversation between an AI code assistant and a user. You will also be given a portion of the conversation you are to focus on. Your job is to simplify the portion of text into abstractions and concepts, such that somebody with little experience in the relevant topic could understand. Wrap your simplification with [BEGIN SIMPLIFICATION] and [END SIMPLIFICATION] markers.
    Here is the conversation excerpt so you have context of the conversation while you generate the simplification. It will be in 
    [BEGIN CONVERSATION]
    {conversation}
    [END CONVERSATION]
    Here is the text that you are to simplify:
    [BEGIN TEXT]
    {input}
    [END TEXT]
    """