# Llama3 70B와 Langchain으로 한국어 Chatbot 만들기

여기에서는 LangChain으로 Llama 70B를 이용해 한국어 챗봇을 만드는 것을 설명합니다. Llama 70B는 대용량 메모리와 GPU를 필요로 하므로, 가벼운 동작테스트를 위해 SageMaker JumpStart를 이용하는것은 부담스러울수 있습니다. 여기에서는 Bedrock API를 이용해 GPU가 포함된 인프라를 구축하지 않고 손쉽게 Llama3 70B로 한국어 챗봇을 구현합니다. 또한, LangChain을 이용해 코드 개발 기간을 단축하고 다른 모델에서 사용된 유사 코드를 가능한 수정없이 사용하고자 합니다. 

## 한국어 Chatbot의 구현

### Llama3 API

여기에서는 LangChain의 [ChatBedrock](https://python.langchain.com/docs/integrations/chat/bedrock/)을 이용해 Llama3 API를 이용합니다. ChatBedrock을 이용하기 위해서 아래와 같이 bedrock-runtime을 위한 boto3_bedrock을 정의하고 LLM Parameter를 지정한 후에 modelId로 "meta.llama3-70b-instruct-v1:0"을 설정합니다.

```python
from langchain_aws import ChatBedrock
bedrock_region = 'us-west-2'
modelId = "meta.llama3-70b-instruct-v1:0"
boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
    config=Config(
        retries = {
            'max_attempts': 30
        }            
    )
)

parameters = {
    "max_gen_len": 1024,  
    "top_p": 0.9, 
    "temperature": 0.1,
}    
chat = ChatBedrock(   
    model_id=modelId,
    client=boto3_bedrock, 
    model_kwargs=parameters,
)
```



### 채팅 이력의 활용

여기에서는 [ConversationBufferWindowMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer_window.ConversationBufferWindowMemory.html)을 이용하여 대화이력을 저장하기 위한 메모리를 정의합니다. 새로운 대화은 아래와 같이 add_user_message()와 add_ai_message()와 같이 저장합니다.

```python
memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=10)

memory_chain.chat_memory.add_user_message(text)
memory_chain.chat_memory.add_ai_message(msg)
```

채팅에서 이전 대화이력은 아래와 같이 로드하여 사용합니다. 

```python
history = memory_chain.load_memory_variables({})["chat_history"]
```




## Prompt formats

[Meta Llama 3 Instract](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/#special-tokens-used-with-meta-llama-3)에 따라 아래와 같은 prompt foramt을 가져야 합니다.

- BOS (beginning of a sentence) token: <|begin_of_text|>

- End of the message in a turn: <|eot_id|>

- Role for a particular message (system, user, assistant): <|start_header_id|>{role}<|end_header_id|>

- EOS token: <|end_of_text|>

상기 조건에 맞는 Prompt는 아래와 같습니다.

```python
def general_conversation(connectionId, requestId, chat, query):
    system = (
"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n
다음은 Human과 AI의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. 
Assistant의 이름은 서연이고, Emoji 없이 가능한 한국어로 답변하세요. 또한, 한자는 한국어로 변환합니다.<|eot_id|>"""
    )
    human = """<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat
        
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)
        
        print('stream: ', stream)        
        usage = stream.response_metadata['usage']
        print('prompt_tokens: ', usage['prompt_tokens'])
        print('completion_tokens: ', usage['completion_tokens'])
        print('total_tokens: ', usage['total_tokens'])
        msg = stream.content
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg
```

### Stream의 처리

LLM의 응답시간은 수초이상이므로 Stream을 사용하여 사용성을 개선합니다. 


