######### https://docs.llamaindex.ai/en/stable/examples/agent/openai_assistant_agent/


# openai RAG code
####################################################
####################################################
####################################################
# import os
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import TextLoader
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# # Prompt the user for their OpenAI API key
# api_key = ""
# os.environ["OPENAI_API_KEY"] = api_key
# print("OPENAI_API_KEY has been set!")

# txt_file_path = 'rag_data/scalexi.txt'
# loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
# data = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# data = text_splitter.split_documents(data)

# embeddings = OpenAIEmbeddings()

# vectorstore = FAISS.from_documents(data, embedding=embeddings)

# llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")

# memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# conversation_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(),
#     memory=memory
# )

# query = "What is ScaleX Innovation?"
# result = conversation_chain({"question": query})
# answer = result["answer"]
# print("ans: ", answer)

# query = "What is the contact information?"
# result = conversation_chain({"question": query})
# answer = result["answer"]
# print("ans: ", answer)

# query = "What are the main activities of ScaleX Innovation. Write is as three bullet points."
# result = conversation_chain({"question": query})
# answer = result["answer"]
# print("ans: ", answer)
####################################################
####################################################
####################################################


# 2 people talking with RAG
#######################################
# import os
# import json
# from typing import Sequence, List
# import nest_asyncio

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import TextLoader
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# from llama_index.llms.openai import OpenAI
# from llama_index.core.llms import ChatMessage
# from llama_index.core.tools import BaseTool, FunctionTool
# from openai.types.chat import ChatCompletionMessageToolCall

# # Set up API key
# api_key = ""
# os.environ["OPENAI_API_KEY"] = api_key
# print("OPENAI_API_KEY has been set!")




# # Load and process text data
# txt_file_path = 'rag_data/scalexi.txt'
# loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
# data = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# data = text_splitter.split_documents(data)

# embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(data, embedding=embeddings)



# txt_file_path2 = 'rag_data/cornell_introduction.txt'
# loader2 = TextLoader(file_path=txt_file_path2, encoding="utf-8")
# data2 = loader2.load()
# text_splitter2 = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# data2 = text_splitter2.split_documents(data2)

# embeddings2 = OpenAIEmbeddings()
# vectorstore2 = FAISS.from_documents(data2, embedding=embeddings2)




# # Configure conversational components
# llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
# memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
# conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), memory=memory)

# conversation_chain2 = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=vectorstore2.as_retriever(), memory=memory)

# # Agent class definition
# class YourOpenAIAgent:
#     def __init__(self, chat_model: ChatOpenAI = llm, memory: ConversationBufferMemory = memory, vector_store=vectorstore):
#         self._chat_model = chat_model
#         self._memory = memory
#         self._vector_store = vector_store

#     def chat(self, message: str) -> str:
#         query = {"question": message}
#         result = conversation_chain(query)
#         answer = result["answer"]
#         return answer

# class YourOpenAIAgent2:
#     def __init__(self, chat_model: ChatOpenAI = llm, memory: ConversationBufferMemory = memory, vector_store=vectorstore2):
#         self._chat_model = chat_model
#         self._memory = memory
#         self._vector_store = vector_store

#     def chat(self, message: str) -> str:
#         query = {"question": message}
#         result = conversation_chain2(query)
#         answer = result["answer"]
#         return answer




# # Initialize agent with tools and llm
# agent = YourOpenAIAgent(chat_model=llm, memory=memory)
# agent2 = YourOpenAIAgent2(chat_model=llm, memory=memory)

# def agents_conversation(start_message, num_turns=5):
#     message = start_message
#     for i in range(num_turns):
#         print(f"Agent 1: {message}")
#         message = agent2.chat(message)
#         print(f"Agent 2: {message}")
#         message = agent.chat(message)

# # Example usage
# agents_conversation("Hi, Where is cornell located? What is ScaleX Innovation?")

#######################################
#######################################
#######################################




# # generate engine and talk to each other and learn from memory
# #######################################
# import os

# # Set up API key
# api_key = ""
# os.environ["OPENAI_API_KEY"] = api_key
# print("OPENAI_API_KEY has been set!")

# from llama_index.agent.openai import OpenAIAssistantAgent
# from llama_index.core import (
#     SimpleDirectoryReader,
#     VectorStoreIndex,
#     StorageContext,
#     load_index_from_storage,
# )

# from llama_index.core.tools import QueryEngineTool, ToolMetadata


# try:
#     storage_context = StorageContext.from_defaults(
#         persist_dir="./storage/lyft"
#     )
#     lyft_index = load_index_from_storage(storage_context)

#     storage_context = StorageContext.from_defaults(
#         persist_dir="./storage/uber"
#     )
#     uber_index = load_index_from_storage(storage_context)

#     index_loaded = True
# except Exception as e:
#     print(e)
#     index_loaded = False

# if not index_loaded:
#     # load data
#     lyft_docs = SimpleDirectoryReader(
#         input_files=["./rag_data/lyft.txt"]
#     ).load_data()
#     uber_docs = SimpleDirectoryReader(
#         input_files=["./rag_data/uber.txt"]
#     ).load_data()

#     # build index
#     lyft_index = VectorStoreIndex.from_documents(lyft_docs)
#     uber_index = VectorStoreIndex.from_documents(uber_docs)

#     # persist index
#     lyft_index.storage_context.persist(persist_dir="./storage/lyft")
#     uber_index.storage_context.persist(persist_dir="./storage/uber")

# lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
# uber_engine = uber_index.as_query_engine(similarity_top_k=3)

# query_engine_tools = [
#     QueryEngineTool(
#         query_engine=lyft_engine,
#         metadata=ToolMetadata(
#             name="lyft_10k",
#             description=(
#                 "Provides information about Lyft financials for year 2021. "
#                 "Use a detailed plain text question as input to the tool."
#             ),
#         ),
#     ),
#     QueryEngineTool(
#         query_engine=uber_engine,
#         metadata=ToolMetadata(
#             name="uber_10k",
#             description=(
#                 "Provides information about Uber financials for year 2021. "
#                 "Use a detailed plain text question as input to the tool."
#             ),
#         ),
#     ),
# ]

# class FinancialAgent:
#     def __init__(self, name, assistant_agent, initial_knowledge):
#         self.name = name
#         self.assistant_agent = assistant_agent
#         self.knowledge = initial_knowledge  # This could be a more complex database

#     def ask_question(self, other_agent, question):
#         print(f"{self.name} asks: {question}")
#         if question in other_agent.knowledge:
#             answer = other_agent.knowledge[question]
#             print(f"{other_agent.name} responds from memory: {answer}")
#         else:
#             answer = other_agent.respond(question)
#             print(f"{other_agent.name} responds after research: {answer}")
#         self.learn(question, answer)

#     def respond(self, question):
#         # Check own knowledge first
#         if question in self.knowledge:
#             return self.knowledge[question]
#         # Use the query tool to find an answer
#         tool = self.assistant_agent._tools[0]  # Assumes the first tool is relevant
#         answer = tool.query_engine.query(question)
#         self.learn(question, answer)
#         return answer

#     def learn(self, question, answer):
#         """Update the agent's knowledge base if the information is new."""
#         if question not in self.knowledge:
#             self.knowledge[question] = answer
#             print(f"{self.name} learns: {answer}")

# # Setup tools for each agent (assuming the tools are initialized as shown previously)
# agent_lyft = OpenAIAssistantAgent.from_new(
#     name="Lyft Analyst",
#     instructions="Analyze Lyft's SEC filings.",
#     tools=[query_engine_tools[0]],  # Lyft tool
#     instructions_prefix="Please use formal language.",
#     verbose=True
# )

# agent_uber = OpenAIAssistantAgent.from_new(
#     name="Uber Analyst",
#     instructions="Analyze Uber's SEC filings.",
#     tools=[query_engine_tools[1]],  # Uber tool
#     instructions_prefix="Please use formal language.",
#     verbose=True
# )

# # Initialize agents with some basic knowledge
# initial_knowledge_lyft = {"Lyft's founding year": "Lyft was founded in 2012."}
# initial_knowledge_uber = {"Uber's founding year": "Uber was founded in 2009."}

# agent1 = FinancialAgent("Lyft Analyst", agent_lyft, initial_knowledge_lyft)
# agent2 = FinancialAgent("Uber Analyst", agent_uber, initial_knowledge_uber)

# # Example conversation
# agent1.respond("hello, can you generate a random story for me?")
# agent1.ask_question(agent2, "What are Uber's revenue and profit figures for 2021?")
# agent2.ask_question(agent1, "What are Lyft's revenue and profit figures for 2021?")
# agent2.ask_question(agent1, "What are Uber's revenue and profit figures for 2021?")
# #######################################
# #######################################
# #######################################



# generate story with RAG
#######################################
import os

# Set up API key
api_key = ""
os.environ["OPENAI_API_KEY"] = api_key
print("OPENAI_API_KEY has been set!")

from llama_index.agent.openai import OpenAIAssistantAgent
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata


try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/egypt"
    )
    story_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/science"
    )
    question_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/bopit"
    )
    bopit_index = load_index_from_storage(storage_context)

    index_loaded = True
except Exception as e:
    print(e)
    index_loaded = False

if not index_loaded:
    # load data
    story_docs = SimpleDirectoryReader(
        input_files=["./rag_data/egypt.txt"]
    ).load_data()
    question_docs = SimpleDirectoryReader(
        input_files=["./rag_data/science.txt"]
    ).load_data()
    bopit_docs = SimpleDirectoryReader(
        input_files=["./rag_data/bopit.txt"]
    ).load_data()

    # build index
    story_index = VectorStoreIndex.from_documents(story_docs)
    question_index = VectorStoreIndex.from_documents(question_docs)
    bopit_index = VectorStoreIndex.from_documents(question_docs)

    # persist index
    story_index.storage_context.persist(persist_dir="./storage/egypt")
    question_index.storage_context.persist(persist_dir="./storage/science")
    bopit_index.storage_context.persist(persist_dir="./storage/bopit")

story_engine = story_index.as_query_engine(similarity_top_k=3)
question_engine = question_index.as_query_engine(similarity_top_k=3)
bopit_engine = question_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=story_engine,
        metadata=ToolMetadata(
            name="egypt_10k",
            description=(
                "This file provides a comprehensive framework for a historical narrative based in Ancient Egypt, utilizing the detailed cultural, political, and scientific context of the civilization. The story is meticulously structured to include key aspects of Egyptian life, offering an immersive experience that is both educational and engaging."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=question_engine,
        metadata=ToolMetadata(
            name="science_10k",
            description=(
                "This file contains a curated assortment of scientific facts, specifically tailored to engage students and enhance their understanding of the natural world. The content spans a diverse array of scientific disciplines, offering a rich resource for educational narratives and interactive learning modules."
                "Each scientific fact can be woven into different parts of a story, such as the introduction to set the scene, challenges where characters must use scientific knowledge to proceed, or conclusions that reflect on the lessons learned."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=bopit_engine,
        metadata=ToolMetadata(
            name="bopit_10k",
            description=(
                "Provides information and example about the structure of the story should be look like. "
                """ 1.Story Architecture: The narrative is structured with an introduction, a series of interactive challenges, an educational interlude, and a conclusion. This structure ensures a steady narrative pace with sections designed to engage and educate young readers effectively.
                    2. Methods of Interaction: The story actively engages children through physical challenges such as twisting, pulling, and passing objects. Each interactive challenge is crafted to advance the story while promoting physical interaction and problem-solving, like twisting a stone wheel to align symbols.
                    3. Ways of Asking Questions: Questions are strategically placed throughout the story to pique curiosity and enhance the learning experience. They vary from direct interactive tasks to educational inquiries about scientific concepts, keeping children engaged and thinking.
                    4. Educational Interlude: An educational segment interrupts the physical challenges to introduce a scientific concept, seamlessly woven into the narrative. This approach enriches the learning experience without diminishing the adventure's excitement, ensuring the story remains engaging and informative."""
            ),
        ),
    ),
]

agentprompt = """You are tasked with creating interactive, educational narratives set within accurate historical contexts. Your stories should combine engaging plotlines with scientific inquiries and dynamic physical interactions, challenging users to problem-solve and think critically. This approach aims to foster a deep understanding and appreciation of history and science."
            Story Architecture:

Segmentation: Structure stories into multiple short sections (100-200 words each), each concluding with a clear interaction point to advance the story.
Interaction Methods: Use buttons, sliders, rotary controls, voice input, and directional buttons for physical interactions.
Interactive Challenges: Ensure challenges are intuitive and fun, enhancing learning and aligning with the storyline.
Decision Points: Provide impactful choices rooted in historical and scientific content, emphasizing cause-effect relationships.
"""
agent = OpenAIAssistantAgent.from_new(
    name="Story teller",
    instructions="This approach aims to foster a deep understanding and appreciation of history and science.",
    tools=query_engine_tools,
    instructions_prefix="Please address the user as Julia.",
    verbose=True,
    run_retrieve_sleep_time=1.0,
)

# Example conversation
response = agent.chat("Generate a advanture story based on Egypt background and mix some scientific problem and more interaction for the user to answer in the story, and help the character conquer the challege")

print(response)
#######################################
#######################################
#######################################


# input speech and state machine
#######################################
# import speech_recognition as sr

# # Define the State base class
# class State:
#     def __init__(self):
#         pass

#     def on_event(self, input):
#         pass

#     def __str__(self):
#         return self.__class__.__name__

# # Define specific states
# class MainMenu(State):
#     def on_event(self, input):
#         if "settings" in input:
#             return SettingsMenu()
#         elif "exit" in input:
#             return ExitState()
#         return self

# class SettingsMenu(State):
#     def on_event(self, input):
#         if "main menu" in input:
#             return MainMenu()
#         elif "exit" in input:
#             return ExitState()
#         return self

# class ExitState(State):
#     def on_event(self, input):
#         return self

#     def __str__(self):
#         return "Exiting"

# # Define the state machine that manages states
# class StateMachine:
#     def __init__(self):
#         self.state = MainMenu()

#     def on_event(self, input):
#         self.state = self.state.on_event(input)
#         print(f"Current State: {self.state}")

# # Speech recognition function
# def recognize_speech_from_mic(recognizer, microphone):
#     with microphone as source:
#         recognizer.adjust_for_ambient_noise(source)
#         print("Listening...")
#         audio = recognizer.listen(source)

#     try:
#         speech_input = recognizer.recognize_google(audio)
#         print(f"You said: {speech_input}")
#         return speech_input.lower()
#     except sr.RequestError:
#         print("API unavailable")
#     except sr.UnknownValueError:
#         print("Unable to recognize speech")
#     return ""

# # Main function to run the state machine with speech input
# def main():
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()
#     machine = StateMachine()

#     while True:
#         input = recognize_speech_from_mic(recognizer, microphone)
#         print(input)
#         if "exit" in input:
#             machine.on_event(input)
#             if isinstance(machine.state, ExitState):
#                 break
#         else:
#             machine.on_event(input)

# if __name__ == "__main__":
#     main()
#######################################
#######################################
#######################################












