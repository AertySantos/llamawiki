import sys
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_PATH = "models/llama-2-70b-chat.Q5_K_M.gguf"

# criar a função que carrega a llama

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 80
n_batch = 1024
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.1,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    n_ctx=4096
)

while True:
    query = input(f"Input Prompt: ")
    if query == 'exit':

        print('Exiting')
        sys.exit()

    llm(query)
