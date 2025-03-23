import logging
import os
import shutil
import subprocess

import torch
from auto_gptq import AutoGPTQForCausalLM
from flask import Flask, jsonify, request
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings

# from langchain.embeddings import HuggingFaceEmbeddings
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
from werkzeug.utils import secure_filename

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
pronto_uri = f"postgresql+psycopg2://{user_name}:{user_pwd}@{host_name}:{port}/{database}"
jdbcHostname = os.getenv('jdbcHostname')
jdbcUsername = os.getenv('jdbcUsername')
jdbcPassword = os.getenv('jdbcPassword')
dbms = os.getenv('dbms')
omnifin_uri = f'mysql+pymysql://{jdbcUsername}:{jdbcPassword}@{jdbcHostname}/{dbms}'

device_type = os.getenv('device_type')
# temporary, just to check questions entered
logging.basicConfig(filename="logs.txt", filemode='a', 
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
)
SHOW_SOURCES = True
logging.info(f"Running on: {device_type}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})

# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# if os.path.exists(PERSIST_DIRECTORY):
#     try:
#         shutil.rmtree(PERSIST_DIRECTORY)
#     except OSError as e:
#         print(f"Error: {e.filename} - {e.strerror}.")
# else:
#     print("The directory does not exist")

# run_langest_commands = ["python", "ingest.py"]
# if DEVICE_TYPE == "cpu":
#     run_langest_commands.append("--device_type")
#     run_langest_commands.append(DEVICE_TYPE)

# result = subprocess.run(run_langest_commands, capture_output=True)
# if result.returncode != 0:
#     raise FileNotFoundError(
#         "No files were found inside SOURCE_DOCUMENTS, please put a starter file inside before starting the API!"
#     )

# load the vectorstore
DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)

RETRIEVER = DB.as_retriever()


# load the LLM for generating Natural Language responses
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
    # logging.info("This action can take a few minutes!")

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
        
    db = SQLDatabase.from_uri(pg_uri, sample_rows_in_table_info=2, schema=schema, include_tables=table_list)
    db._schema=None
    # Create db chain
    if pg_uri == pronto_uri:
        QUERY = """
        Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
        Use the following format:

        Question: Question here
        SQLQuery: SQL Query to run
        SQLResult: Result of the SQLQuery
        Answer: Final answer here

        {question}
        """
    else:
        QUERY = """
        Given an input question, first create a syntactically correct mysql query to run, then look at the results of the query and return the answer.
        Use the following format:

        Question: Question here
        SQLQuery: SQL Query to run
        SQLResult: Result of the SQLQuery
        Answer: Final answer here

        {question}
        """
    # Setup the database chain
    # use return_direct=True to directly return the output of the SQL query without any additional formatting
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True, return_intermediate_steps=True)
    # question = QUERY.format(question=question)
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

if device_type != "cuda":
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
    LLM = load_model(device_type, model_id=model_id, model_basename=model_basename)
else:
    LLM = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

QA = RetrievalQA.from_chain_type(
    llm=LLM, chain_type="stuff", retriever=RETRIEVER, return_source_documents=SHOW_SOURCES
)

app = Flask(__name__)


@app.route("/api/delete_source", methods=["GET"])
def delete_source_route():
    folder_name = "SOURCE_DOCUMENTS_EXTERNAL"

    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)

    return jsonify({"message": f"Folder '{folder_name}' successfully deleted and recreated."})


@app.route("/api/save_document", methods=["GET", "POST"])
def save_document_route():
    if "document" not in request.files:
        return "No document part", 400
    file = request.files["document"]
    if file.filename == "":
        return "No selected file", 400
    if file:
        filename = secure_filename(file.filename)
        folder_path = "SOURCE_DOCUMENTS_EXTERNAL"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, filename)
        file.save(file_path)
        return "File saved successfully", 200


@app.route("/api/run_ingest", methods=["GET"])
def run_ingest_route():
    global DB
    global RETRIEVER
    global QA
    PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB1"
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")
        else:
            print("The directory does not exist")

        run_langest_commands = ["python", "ingest_external.py"]
        if device_type != "cuda":
            run_langest_commands.append("--device_type")
            run_langest_commands.append(device_type)
            
        result = subprocess.run(run_langest_commands, capture_output=True)
        if result.returncode != 0:
            return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500
        # load the vectorstore
        CHROMA_SETTINGS = Settings(
            chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
        )
        DB = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDINGS,
            client_settings=CHROMA_SETTINGS,
        )
        RETRIEVER = DB.as_retriever()

        QA = RetrievalQA.from_chain_type(
            llm=LLM, chain_type="stuff", retriever=RETRIEVER, return_source_documents=SHOW_SOURCES
        )
        return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500


@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
    user_prompt = request.form.get("user_prompt")
    logging.info(f'User Prompt: {user_prompt}')
    if user_prompt:
        # print(f'User Prompt: {user_prompt}')
        # Get the answer from the chain
        if  user_prompt.startswith('E:'):
            # load the vectorstore            
            PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB1"
            CHROMA_SETTINGS = Settings(
                chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
            )
            DB1 = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=EMBEDDINGS,
                client_settings=CHROMA_SETTINGS,
            )
            RETRIEVER = DB1.as_retriever()

            QA = RetrievalQA.from_chain_type(
                llm=LLM, chain_type="stuff", retriever=RETRIEVER, return_source_documents=SHOW_SOURCES
            )
            res = QA(user_prompt)
            answer, docs = res["result"], res["source_documents"]

            prompt_response_dict = {
                "Prompt": user_prompt,
                "Answer": answer,
            }

            prompt_response_dict["Sources"] = []
            for document in docs:
                prompt_response_dict["Sources"].append(
                    (os.path.basename(str(document.metadata["source"])), str(document.page_content))
                )

            return jsonify(prompt_response_dict), 200
        else:
            # load the vectorstore            
            PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"
            CHROMA_SETTINGS = Settings(
                chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
            )
            DB = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=EMBEDDINGS,
                client_settings=CHROMA_SETTINGS,
            )
            RETRIEVER = DB.as_retriever()
            QA = RetrievalQA.from_chain_type(
                llm=LLM, chain_type="stuff", retriever=RETRIEVER, return_source_documents=SHOW_SOURCES
            )
            res = QA(user_prompt)
            answer, docs = res["result"], res["source_documents"]

            prompt_response_dict = {
                "Prompt": user_prompt,
                "Answer": answer,
            }

            prompt_response_dict["Sources"] = []
            for document in docs:
                prompt_response_dict["Sources"].append(
                    (os.path.basename(str(document.metadata["source"])), str(document.page_content))
                )

            return jsonify(prompt_response_dict), 200           
    else:
        return "No user prompt received", 400
    
@app.route("/api/sql_route", methods=["GET", "POST"])
def sql_route():
    global LLM
    user_prompt = request.form.get("user_prompt")
    logging.info(f'User Prompt: {user_prompt}')
    data_list = user_prompt.split(";")
    user_prompt = data_list[0]
    try:
        db_name = data_list[-3].replace(" ", "")
    except:
        db_name = "omnifin"
    try:
        schema = data_list[-2].replace(" ", "")
    except:
        schema = "beefin_uat"
    tables = data_list[-1].replace(" ", "")
    if db_name.lower() == 'pronto':
        pg_uri = pronto_uri
    else:
        pg_uri = omnifin_uri
    if user_prompt:
        # print(f'User Prompt: {user_prompt}')
        # Get the answer from the chain
        result = sql_mode(LLM, pg_uri, schema, [tables], user_prompt)
        answer, docs = result["intermediate_steps"][-1], result["intermediate_steps"][1]

        prompt_response_dict = {
            "Prompt": user_prompt,
            "Answer": answer,
        }

        prompt_response_dict["Sources"] = [("SQL Query", str(docs))]

        return jsonify(prompt_response_dict), 200
    else:
        return "No user prompt received", 400
    
@app.route("/api/web_route", methods=["GET", "POST"])
def web_route():
    global LLM
    global EMBEDDINGS
    user_prompt = request.form.get("user_prompt")
    logging.info(f'User Prompt: {user_prompt}')
    data_list = user_prompt.split(";")
    user_prompt = data_list[0]
    url = data_list[1]
    if user_prompt:
        # print(f'User Prompt: {user_prompt}')
        # Get the answer from the chain
        res = url_mode(LLM, user_prompt, url, EMBEDDINGS)
        answer, docs = res["result"], res["source_documents"]

        prompt_response_dict = {
            "Prompt": user_prompt,
            "Answer": answer,
        }

        prompt_response_dict["Sources"] = []
        for document in docs:
            prompt_response_dict["Sources"].append(
                (os.path.basename(str(document.metadata["source"])), str(document.page_content))
            )

        return jsonify(prompt_response_dict), 200
    else:
        return "No user prompt received", 400


if __name__ == "__main__":
    logging.basicConfig(filename="logs.txt", filemode='a',
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    app.run(debug=True, port=5110)
