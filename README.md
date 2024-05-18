# Llama3 70B와 Langchain으로 한국어 챗봇 만들기

여기에서는 LangChain으로 Llama 70B를 이용해 한국어 챗봇을 만들니다. Llama 70B는 대용량 메모리와 GPU를 필요로 하므로, 가벼운 동작테스트를 위해 SageMaker JumpStart를 이용하는것은 부담스러울수 있습니다. 여기에서는 Bedrock API를 이용해 GPU가 포함된 인프라를 구축하지 않고 손쉽게 Llama3 70B로 한국어 챗봇을 만들어보고자 합니다. 또한, LangChain을 이용해 코드 개발 기간을 단축하고 다른 모델에서 사용된 유사 코드를 가능한 수정없이 사용하고자 합니다. 

## 한국어 챗봇의 구현

### Llama3 API

여기에서는 LangChain의 [ChatBedrock](https://python.langchain.com/docs/integrations/chat/bedrock/)을 이용해 Llama3 API를 이용합니다. 

### Stream의 처리

LLM의 응답시간은 수초이상이므로 Stream을 사용하여 사용성을 개선합니다. 


### 채팅 이력의 활용



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
```
