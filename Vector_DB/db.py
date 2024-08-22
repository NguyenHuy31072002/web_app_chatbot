from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import SentenceTransformerEmbeddings
# Khai bao bien
pdf_data_path = "C:/Users/PC/Desktop/Chatbot/chatbot/data"
vector_db_path = "vectorstores/db_faiss"
txt_data_path = "C:/Users/PC/Desktop/chatGemini/Gemini-AI-chatbot/data"

# Ham 1. Tao ra vector DB tu 1 doan text
def create_db_from_text():
    raw_text = """Chủ tịch Hồ Chí Minh, người lãnh tụ vĩ đại của dân tộc Việt Nam, không chỉ là một nhà chính trị lỗi lạc mà còn là biểu tượng của
            lòng nhân ái và tinh thần hy sinh. Sinh ra trong hoàn cảnh nghèo khó, ông đã vượt qua mọi thử thách để trở thành một nhà lãnh đạo
            sáng suốt, kiên định. Với trí tuệ sáng suốt và trái tim nhân hậu, Hồ Chí Minh đã lãnh đạo phong trào đấu tranh giành độc lập và
            xây dựng nền tảng cho một đất nước tự do, hòa bình. Tinh thần kiên cường và tầm nhìn chiến lược của ông đã dẫn dắt nhân dân Việt Nam
            vượt qua những giai đoạn gian nan, từ cuộc đấu tranh chống thực dân Pháp đến chiến thắng trong cuộc kháng chiến chống Mỹ.
            Sự hy sinh và cống hiến của Chủ tịch Hồ Chí Minh đã để lại di sản quý giá, làm nên một trang sử hào hùng trong lịch sử dân tộc,
            và vẫn mãi là nguồn cảm hứng cho các thế hệ mai sau trong công cuộc xây dựng và phát triển đất nước."""

    # Chia nho van ban
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len

    )

    chunks = text_splitter.split_text(raw_text)

    # Embeding
    embeddings = SentenceTransformerEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code":True})



    # Dua vao Faiss Vector DB
    db = FAISS.from_texts(texts=chunks, embedding=embeddings)
    db.save_local(vector_db_path)
    return db


def create_db_from_files():
    """_summary_
    - Doc file data pdf
    - Xu ly va chia data thanh cac doan nho
    - goi mo hinh embeding data
    - luu vector_data vao FAISS
    Returns:
        _type_: db
    """
    # Khai bao loader de quet toan bo thu muc data
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embeddings = SentenceTransformerEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"trust_remote_code":True})
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vector_db_path)
    return db

from langchain_community.document_loaders.text import TextLoader

class CustomTextLoader(TextLoader):
    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        super().__init__(file_path)
        self.encoding = encoding

    def _load(self):
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            text = f.read()
        return text

def create_db_from_files_txt():
    """Đọc tất cả các file data dưới dạng txt
        - Chia nhỏ văn bản đọc vào thành các đoạn nhỏ.
        - Sử dụng mô hình Embeddings để chuyển hóa các đoạn nhỏ đấy
        thành vector và lưu trữ vào FAISS(Facebook AI)
    
    """
    # Khai báo loader để quét toàn bộ thư mục data
    loader = DirectoryLoader(txt_data_path, glob="*.txt", loader_cls=lambda file_path: CustomTextLoader(file_path, encoding='utf-8'))
    documents = loader.load()

    # Chia đoạn văn bản với kích thước 1024 và chồng lặp 50
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embedding
    embeddings = SentenceTransformerEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"trust_remote_code": True})
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vector_db_path)
    return db

create_db_from_files_txt()

