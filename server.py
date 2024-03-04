from flask import Flask, request, jsonify
from tempfile import NamedTemporaryFile
import os
from llama_index.core import VectorStoreIndex,ServiceContext
from llama_index.readers.file.docs import PDFReader
from llama_index.llms.openai import OpenAI
from pathlib import Path
from typing import List
from pydantic import BaseModel
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

# DEBUG
#import logging
#import sys
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

app = Flask(__name__)

# Define HTTP route, method, and handler.
#################################
@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():

    # Read file from upload
    #################################

    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded file to temp directory
    #################################
    if file and allowed_file(file.filename):
        with NamedTemporaryFile(delete=False) as tmp:
            # save uploaded file
            file.save(tmp.name)
            
        # Read file and create index
        #################################

        loader = PDFReader()
        document_path = Path(tmp.name)
        documents = loader.load_data(file=document_path)
        index = VectorStoreIndex.from_documents(documents)

        # Define output schema
        #################################
        class Deliverable(BaseModel):
            """Data model for deliverables"""
            deliverable: str
            deadline: str

        class Result(BaseModel):
            """Data model for results"""
            rate: str
            deliverables: List[Deliverable]
            paymentTimeline: str
        
        # Define query engine configurations
        #################################
        llm = OpenAI(model="gpt-3.5-turbo")

        _service_context = ServiceContext.from_defaults(
            llm=llm,
            chunk_size=1024,
            chunk_overlap=int(200),
        )

        # Text QA Prompt
        chat_text_qa_msgs = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "Always answer the question, even if the context isn't helpful."
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, "
                    "answer the question: {query_str}\n"
                ),
            ),
        ]
        text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

        # Refine Prompt
        chat_refine_msgs = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "Always answer the question, even if the context isn't helpful."
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "We have the opportunity to refine the original answer "
                    "(only if needed) with some more context below.\n"
                    "------------\n"
                    "{context_msg}\n"
                    "------------\n"
                    "Given the new context, refine the original answer to better "
                    "answer the question: {query_str}. "
                    "If the context isn't useful, output the original answer again.\n"
                    "Original Answer: {existing_answer}"
                ),
            ),
        ]
        refine_template = ChatPromptTemplate(chat_refine_msgs)

        # Configure query engine
        #################################
        query_engine = index.as_query_engine(
            output_cls=Result,
            response_mode="compact",
            service_context=_service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
        )

        # Query the index and get structured response
        #################################
        response = query_engine.query("what is the fee, what are the deliverables and their timelines, and what is the payment timeline?")
        
        # Make sure to delete temp file created at upload time
        #################################
        os.unlink(tmp.name)

        # Debug
        #prompts_dict = query_engine.get_prompts()
        #print(prompts_dict)
        #print(response.response.model_dump())

        return dict(response.response.model_dump())
    else:
        return jsonify({'error': 'Invalid file type'}), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['pdf']

if __name__ == '__main__':
    app.run(debug=True, port=8888)