import os
import chainlit as cl
from backend.main import Rag

# Initialize the RAG system
rag = Rag(
    pdf_directory="./data",
    persist_directory="./chroma_db",
    huggingface_api_key=os.environ.get("HUGGINGFACE_API_KEY")
)

# @cl.on_chat_start

# async def on_chat_start():
#     """Khi bắt đầu chat, gửi lời chào và hiển thị các hành động."""
#     await cl.Message(content="Hi! What can I help you with today?").send()

#     # Add action buttons for users
#     actions = [
#         cl.Action(name="create_index", value="create", label="Tạo chỉ mục"),
#         cl.Action(name="on_message", value="ask", label="Hỏi câu hỏi")
#     ]
#     await cl.Message(content="Chọn hành động:", actions=actions).send()

# @cl.action_callback("create_index")
# async def create_index(action):
#     """Tạo chỉ mục từ tài liệu PDF."""
#     await cl.Message(content="Đang tạo chỉ mục vector...").send()
#     try:
#         documents = rag.load_and_split_documents()
#         rag.create_chroma(documents)
#         await cl.Message(content="✅ Đã tạo chỉ mục vector thành công!").send()
#     except Exception as e:
#         await cl.Message(content=f"❌ Lỗi khi tạo chỉ mục: {str(e)}").send()

@cl.on_message
async def on_message(message: cl.Message):
    try:
        question = message.content
        response = rag.generate_answer(question)  # Gọi LLM để sinh câu trả lời
        
        await cl.Message(content=response).send()

    except Exception as e:
        await cl.Message(content=f"❌ Đã xảy ra lỗi: {str(e)}").send()
# @cl.on_file_upload(accept=["application/pdf"])
# async def on_file_upload(file: cl.File):
#     """Xử lý file PDF được tải lên."""
#     await cl.Message(content=f"📂 Đang xử lý file '{file.name}'...").send()

#     os.makedirs("./data", exist_ok=True)  # Đảm bảo thư mục tồn tại
#     file_path = os.path.join("./data", file.name)
    
#     with open(file_path, "wb") as f:
#         f.write(await file.get_bytes())

#     await cl.Message(content=f"✅ Đã tải lên file '{file.name}' thành công!").send()
