# from flask import Flask, render_template, request, redirect, url_for
# import google.generativeai as genai

# model = genai.GenerativeModel('gemini-1.5-flash')

# import os
# my_api_key_gemini = 'AIzaSyCZh9YLA0zKAlm8NnKKucNd2u-VJNI59ew'

# genai.configure(api_key=my_api_key_gemini)

# app = Flask(__name__)

# # Define your 404 error handler to redirect to the index page
# @app.errorhandler(404)
# def page_not_found(e):
#     return redirect(url_for('index'))

# @app.route('/', methods=['POST', 'GET'])
# def index():
#     if request.method == 'POST':
#         try:
#             prompt = request.form['prompt']
#             question = prompt

#             response = model.generate_content(question)

#             if response.text:
#                 return response.text
#             else:
#                 return "Sorry, but I think Gemini didn't want to answer that!"
#         except Exception as e:
#             return "Sorry, but Gemini didn't want to answer that!"

#     return render_template('index.html', **locals())

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
from dotenv import load_dotenv
from flask import Flask, render_template, redirect, url_for



# Load biến môi trường từ file .env
load_dotenv()

model = genai.GenerativeModel('gemini-1.5-flash')

my_api_key_gemini = os.getenv("GOOGLE_GENAI_API_KEY")

genai.configure(api_key=my_api_key_gemini)

# Initialize GenerativeModel
model = genai.GenerativeModel('gemini-1.5-flash')

# Configure path to VectorDB
vector_db_path = "C:/Users/PC/Desktop/chatGemini/Gemini-AI-chatbot/Vector_DB/vectorstores/db_faiss"

# Read from VectorDB
def read_vectors_db():
    embeddings = SentenceTransformerEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"trust_remote_code": True})
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    return db

# Create prompt template
def create_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Create QA chain
def create_qa_chain(prompt, db):
    def custom_llm(query, context):
        full_prompt = prompt.format(context=context, question=query)
        response = model.generate_content(full_prompt)
        if "Tôi không biết" in response.text:
            response = model.generate_content(query)
        return response.text

    class CustomRetrievalQA:
        def __init__(self, retriever, prompt):
            self.retriever = retriever
            self.prompt = prompt
        
        def invoke(self, inputs):
            query = inputs["query"]
            docs = self.retriever.get_relevant_documents(query)
            context = " ".join([doc.page_content for doc in docs])
            answer = custom_llm(query, context)
            links = [doc.metadata['source'] for doc in docs]
            return {"answer": answer, "links": links}
    
    retriever = db.as_retriever(search_kwargs={"k": 1}, max_tokens_limit=6000)
    return CustomRetrievalQA(retriever, prompt)

app = Flask(__name__)

# Define your 404 error handler to redirect to the index page
@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            db=read_vectors_db()
            template = """Sử dụng thông tin sau đây để trả lời câu hỏi một cách chính xác và ngắn gọn. Không thêm bớt, chỉnh sửa hoặc diễn giải lại thông tin. Nếu bạn không biết câu trả lời, hãy nói 'Tôi không biết'.

            Thông tin:
            {context}

            Câu hỏi:
            {question}
            Trả lời giống hệt dữ liệu mà không thêm bớt:"""
            prompt = create_prompt(template)
            
            prompt_1 = request.form['prompt']
            question = prompt_1
            llm_chain = create_qa_chain(prompt, db)

            response = llm_chain.invoke({"query": question})
            return jsonify(response["answer"])
        except Exception as e:
            return "Sorry, but Gemini didn't want to answer that!"
    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, render_template, request, jsonify
# import google.generativeai as genai
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# import os
# from dotenv import load_dotenv
# from flask import Flask, render_template, redirect, url_for

# # Load environment variables
# load_dotenv()

# # Get API key from environment variables
# api_key = os.getenv("GOOGLE_GENAI_API_KEY")

# # Configure Google Generative AI
# genai.configure(api_key=api_key)

# # Initialize GenerativeModel
# model = genai.GenerativeModel('gemini-1.5-flash')

# # Configure path to VectorDB
# vector_db_path = "C:/Users/PC/Desktop/Chatbot/chatbot/src/VectorDB/vectorstores/db_faiss"

# # Read from VectorDB
# def read_vectors_db():
#     embeddings = SentenceTransformerEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"trust_remote_code": True})
#     db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
#     return db

# # Create prompt template
# def create_prompt(template):
#     return PromptTemplate(template=template, input_variables=["context", "question"])

# # Create QA chain
# def create_qa_chain(prompt, db):
#     def custom_llm(query, context):
#         full_prompt = prompt.format(context=context, question=query)
#         response = model.generate_content(full_prompt)
#         if "Tôi không biết" in response.text:
#             response = model.generate_content(query)
#         return response.text

#     class CustomRetrievalQA:
#         def __init__(self, retriever, prompt):
#             self.retriever = retriever
#             self.prompt = prompt
        
#         def invoke(self, inputs):
#             query = inputs["query"]
#             docs = self.retriever.get_relevant_documents(query)
#             context = " ".join([doc.page_content for doc in docs])
#             answer = custom_llm(query, context)
#             links = [doc.metadata['source'] for doc in docs]
#             return {"answer": answer, "links": links}
    
#     retriever = db.as_retriever(search_kwargs={"k": 1}, max_tokens_limit=6000)
#     return CustomRetrievalQA(retriever, prompt)

# app = Flask(__name__)

# # Define 404 error handler
# @app.errorhandler(404)
# def page_not_found(e):
#     return redirect(url_for('index'))

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         try:
#             question = request.form['prompt']
#             db = read_vectors_db()
#             template = """Sử dụng thông tin sau đây để trả lời câu hỏi một cách chính xác và ngắn gọn. Không thêm bớt, chỉnh sửa hoặc diễn giải lại thông tin. Nếu bạn không biết câu trả lời, hãy nói 'Tôi không biết'.

#             Thông tin:
#             {context}

#             Câu hỏi:
#             {question}
#             Trả lời giống hệt dữ liệu mà không thêm bớt:"""
#             prompt = create_prompt(template)
#             llm_chain = create_qa_chain(prompt, db)

#             response = llm_chain.invoke({"query": question})
#             return jsonify(response["answer"])
#         except Exception as e:
#             return jsonify("Sorry, but there was an error processing your request.")
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)


