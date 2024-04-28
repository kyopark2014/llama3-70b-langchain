# llama3-70b-langchain
It shows how to deploy a chatbot using Llama3 70B based on LangChain.

[ChatBedrock](https://python.langchain.com/docs/integrations/chat/bedrock/)


## Prompt formats

[Meta Llama 3](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/#special-tokens-used-with-meta-llama-3)에 따라면 아래와 같은 prompt를 가져야 합니다.

- BOS (beginning of a sentence) token: <|begin_of_text|>

- End of the message in a turn: <|eot_id|>

- role for a particular message (system, user, assistant): <|start_header_id|>{role}<|end_header_id|>

- EOS token: <|end_of_text|>


예는 아래와 같습니다.

```text
<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>\n\nAlways answer without emojis in Korean<|eot_id|>
        <|start_header_id|>user<|end_header_id|>\n\n"{text}"<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n"""
<|end_of_text|>
```
