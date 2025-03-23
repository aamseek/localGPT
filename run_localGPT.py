import logging

import click
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, ROOT_DIRECTORY
from chromadb.config import Settings

from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
# from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv
load_dotenv()
user_name = os.getenv('user_name')
user_pwd = os.getenv('user_pwd')
llm_type = os.getenv('llm_type')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
gcp_project = os.getenv('gcp_project')
host_name = os.getenv('host_name')
port = 5432
database = 'rain'
pg_uri = f"postgresql+psycopg2://{user_name}:{user_pwd}@{host_name}:{port}/{database}"

def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if device_type.lower() in ["cpu", "mps"]:
            logging.info("Using Llamacpp for quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            if device_type.lower() == "mps":
                return LlamaCpp(
                    model_path=model_path,
                    n_ctx=2048,
                    max_tokens=2048,
                    temperature=0,
                    repeat_penalty=1.15,
                    n_gpu_layers=1000,
                )
            return LlamaCpp(model_path=model_path, n_ctx=2048, max_tokens=2048, temperature=0, repeat_penalty=1.15)

        else:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm

def sql_mode(llm, pg_uri, schema, table_list, question) -> str:
    db = SQLDatabase.from_uri(pg_uri, schema=schema, include_tables=table_list)

    # Create db chain
    QUERY = """
    {question}
    """
    # Setup the database chain
    # use return_direct=True to directly return the output of the SQL query without any additional formatting
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True, return_intermediate_steps=True)

    question = QUERY.format(question=question)
    return db_chain(question)

def url_mode(llm, query, url, embeddings, file_type=None) -> str:
    loader = UnstructuredURLLoader(urls=[url])
    loaded_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunked_docs = text_splitter.split_documents(loaded_docs)
    PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB1"
    CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)
    db1 = Chroma.from_documents(
        chunked_docs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    # db1.persist()
    # db1 = None
    retriever = db1.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    res = qa(query)
    return res


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--mode",
    default="doc",
    type=click.Choice(
        [
            "doc",
            "sql",
            "web",
        ],
    ),
    help="Run mode can be doc, sql or web. (Default is doc)",
)
def main(device_type, show_sources, mode):
    """
    This function implements the information retrieval task.


    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})

    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # load the LLM for generating Natural Language responses

    # for HF models
    model_id = "TheBloke/vicuna-7B-1.1-HF"
    model_basename = None
    # model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
    # model_id = "TheBloke/guanaco-7B-HF"
    # model_id = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
    # alongside will 100% create OOM on 24GB cards.
    # llm = load_model(device_type, model_id=model_id)

    if device_type == "cuda":
        # for GPTQ (quantized) models
        # model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
        # model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
        # model_id = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
        # model_basename = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" # Requires
        # ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
        # model_id = "TheBloke/wizardLM-7B-GPTQ"
        # model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
        # model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
        # model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
        # model_id = "TheBloke/Nous-Hermes-Llama2-GPTQ"
        model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
        model_basename = "model.safetensors"  
    else:
        # for GGML (quantized cpu+gpu+mps) models - check if they support llama.cpp
        # model_id = "TheBloke/wizard-vicuna-13B-GGML"
        # model_basename = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
        # model_basename = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
        # model_basename = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
        # model_id = "TheBloke/orca_mini_3B-GGML"
        # model_basename = "orca-mini-3b.ggmlv3.q4_0.bin"
        # model_id = "TheBloke/orca_mini_v2_7B-GGML"
        # model_basename = "orca-mini-v2_7b.ggmlv3.q3_K_M.bin"
        # model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-GGML"
        # model_basename = "Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_K_S.bin"
        # model_basename = "Wizard-Vicuna-7B-Uncensored.ggmlv3.q3_K_S.bin"
        # model_id = "TheBloke/Llama-2-7B-Chat-GGML"
        # model_basename = "llama-2-7b-chat.ggmlv3.q3_K_S.bin"
        model_id = "TheBloke/Nous-Hermes-Llama2-GGML"
        model_basename = "nous-hermes-llama2-13b.ggmlv3.q3_K_S.bin"

    if llm_type == "open_source":
        llm = load_model(device_type, model_id=model_id, model_basename=model_basename)
    else:
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    if mode == "doc":
        # load the vectorstore
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        # Interactive questions and answers
        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break
            # Get the answer from the chain
            res = qa(query)
            answer, docs = res["result"], res["source_documents"]

            # Print the result
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            if show_sources:  # this is a flag that you can set to disable showing answers.
                # # Print the relevant sources used for the answer
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
                    print(document.page_content)
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
    elif mode == "sql":
        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break        
            schema = input("Enter dataset: ")
            tables = input("Enter tables: ")
            table_list = tables.replace(" ", "").split(",")
            result = sql_mode(llm, pg_uri, schema, table_list, query)
            print(result["intermediate_steps"][-1])
    elif mode == "external_doc":
        PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB1"
        CHROMA_SETTINGS = Settings(
        chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
    )    
        # load the vectorstore
        db1 = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
        retriever = db1.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        # Interactive questions and answers
        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break
            # Get the answer from the chain
            res = qa(query)
            answer, docs = res["result"], res["source_documents"]

            # Print the result
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            if show_sources:  # this is a flag that you can set to disable showing answers.
                # # Print the relevant sources used for the answer
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
                    print(document.page_content)
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
    
    else:
        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break   
            url = input("\nEnter URL: ")
            res = url_mode(llm, query, url, embeddings)
            answer, docs = res["result"], res["source_documents"]

            # Print the result
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            if show_sources:  # this is a flag that you can set to disable showing answers.
                # # Print the relevant sources used for the answer
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
                    print(document.page_content)
                print("----------------------------------SOURCE DOCUMENTS---------------------------")



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
