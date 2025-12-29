import gradio as gr
from app_engine import DualModelEngine
from PIL import Image

# 1. Khởi tạo Engine
try:
    engine = DualModelEngine(model_dir="models")
except Exception as e:
    print(f"Lỗi khởi tạo: {e}")
    print("Hãy đảm bảo cấu trúc thư mục 'models/blip_ver' và 'models/clip_ver' đúng.")
    exit()

# 2. Hàm xử lý logic UI
def magic_caption(image, model_name):
    if image is None:
        return "⚠️ Vui lòng tải ảnh lên."
    
    try:
        key = model_name.split()[0] 
        caption = engine.predict(image, model_choice=key)
        
        return f"[{model_name}]: {caption}"
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

# 3. Thiết kế giao diện
with gr.Blocks(title="KTVIC Dual System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🖼️ Hệ Thống Captioning Đa Model (KTVIC)
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Ảnh đầu vào", height=300)
            
            # Dropdown chọn model
            model_selector = gr.Radio(
                choices=["CLIP ", "BLIP "],  
                value="CLIP ", 
                label="Chọn Model AI"
            )
            
            btn = gr.Button("✨ Sinh Caption", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Kết quả mô tả", 
                lines=4, 
                placeholder="Kết quả sẽ hiện ở đây..."
            )
            
    # Xử lý sự kiện
    # Khi bấm nút, gửi ảnh + model đã chọn vào hàm magic_caption
    btn.click(fn=magic_caption, inputs=[input_img, model_selector], outputs=output_text)

# 4. Chạy App
if __name__ == "__main__":
    demo.launch(share=True)