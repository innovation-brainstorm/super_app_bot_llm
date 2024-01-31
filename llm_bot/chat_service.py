from langchain.chains import LLMChain
from langchain_openai  import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from entities import ChatRequest

class OpenAIChat(object):

    def __init__(self,api_key:str):
        self.chat_model = ChatOpenAI(openai_api_key=api_key,model_name="gpt-3.5-turbo",
                                     temperature=0,max_new_tokens=256)
        
        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True,k=5)

        prompt = ChatPromptTemplate(
                        messages=[
                            SystemMessagePromptTemplate.from_template(
                                "Don’t make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
                            ),
  
                            MessagesPlaceholder(variable_name="chat_history"),
                            HumanMessagePromptTemplate.from_template("{question}")
                        ]
                    )
        
        self.chain = prompt | self.chat_model

        chain_with_history = RunnableWithMessageHistory(
                                self.chain,
                                lambda session_id: BaseChatMessageHistory(),
                                input_messages_key="question",
                                history_messages_key="chat_history",
                            )

    def chat(self,chat_request:ChatRequest):
        question=chat_request.message
        llm.invoke("what's 5 times three", tools=[convert_to_openai_tool(multiply)])