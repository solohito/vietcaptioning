import torch
import torch.nn as nn
import clip
import json
import pickle
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================================================
# 1. ĐỊNH NGHĨA DECODER (CẬP NHẬT ĐỂ KHỚP VỚI TRAINING MỚI)
# =========================================================
class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, feature_dim):
        """
        feature_dim: Output gốc của Encoder (CLIP=512, BLIP=768)
        embed_dim: Kích thước Embedding (thường là 512 như config)
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # --- CÁC LỚP MỚI (Project + BatchNorm) ---
        # Đây là lý do gây lỗi "Unexpected key" ở code cũ
        self.project = nn.Linear(feature_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        
        # Input của GRU = Embed_Dim (từ) + Embed_Dim (ảnh đã chiếu)
        # Ví dụ: 512 + 512 = 1024 (Khớp với lỗi size mismatch 1024 của bạn)
        self.gru = nn.GRU(embed_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # Hàm này giữ lại để load weights không lỗi, dù App không dùng forward trực tiếp
        img_features = self.project(features)
        img_features = self.bn(img_features)
        embeddings = self.embed(captions[:, :-1])
        img_features_expanded = img_features.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        gru_input = torch.cat((img_features_expanded, embeddings), dim=2)
        hiddens, _ = self.gru(gru_input)
        return self.fc(hiddens)

    def generate(self, features, vocab_stoi, vocab_itos, max_len=30, device='cpu'):
        # Inference Logic
        batch_size = features.size(0)
        tokens = torch.tensor([[vocab_stoi["<bos>"]]] * batch_size).to(device)
        
        # 1. Chiếu feature ảnh về đúng kích thước
        img_features = self.project(features)
        img_features = self.bn(img_features)
        
        # 2. Chuẩn bị feature để nối vào từng bước [B, 1, embed_dim]
        feature_step = img_features.unsqueeze(1)
        
        hidden = None
        generated_tokens = []
        curr_input = self.embed(tokens)

        for _ in range(max_len):
            # Nối feature ảnh vào input từ hiện tại
            gru_input = torch.cat((feature_step, curr_input), dim=2)
            
            output, hidden = self.gru(gru_input, hidden)
            output = self.fc(output.squeeze(1))
            
            predicted = output.argmax(1).unsqueeze(1)
            generated_tokens.append(predicted)
            
            curr_input = self.embed(predicted)
            
            if (predicted == vocab_stoi["<eos>"]).all(): 
                break
                
        return torch.cat(generated_tokens, 1)

# =========================================================
# 2. CLASS QUẢN LÝ DUAL MODEL
# =========================================================
class DualModelEngine:
    def __init__(self, model_dir="models", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔄 Khởi tạo Engine trên thiết bị: {self.device}")
        
        self.model_dir = model_dir
        self.loaded_models = {} 
        
        print("⏳ Đang load Base Models (CLIP & BLIP)...")
        # Load CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        # Load BLIP
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_vision = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").vision_model.to(self.device)
        self.blip_vision.eval()
        print("✅ Base Models đã sẵn sàng!")

    def _load_decoder(self, model_type):
        if model_type in self.loaded_models:
            return self.loaded_models[model_type]
        
        folder = "blip_ver" if model_type == "BLIP" else "clip_ver"
        path = f"{self.model_dir}/{folder}"
        
        print(f"📥 Đang load Decoder riêng cho {model_type}...")
        
        try:
            with open(f"{path}/config.json", "r") as f: config = json.load(f)
            with open(f"{path}/vocab.pkl", "rb") as f: vocab = pickle.load(f)
        except FileNotFoundError:
            raise Exception(f"Thiếu file config.json hoặc vocab.pkl trong {path}")

        # Lấy tham số từ config
        embed_size = config.get("embed_size", 512) 
        hidden_size = config.get("hidden_size", 512)
        vocab_size = config.get("vocab_size", len(vocab["stoi"]))
        
        # Xử lý Feature Dim
        # Nếu dùng CLIP thì output gốc là 512, BLIP là 768
        # Nếu trong config file có lưu "feature_dim" thì lấy, không thì đoán dựa trên model_type
        if "feature_dim" in config:
            feature_dim = config["feature_dim"]
        else:
            feature_dim = 768 if model_type == "BLIP" else 512

        # Init Decoder với kiến trúc mới
        decoder = GRUDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_size,
            hidden_dim=hidden_size,
            feature_dim=feature_dim
        ).to(self.device)
        
        try:
            decoder.load_state_dict(torch.load(f"{path}/model.pth", map_location=self.device))
        except RuntimeError as e:
            print(f"❌ LỖI LOAD MODEL {model_type}: Sai lệch kiến trúc!")
            print("Chi tiết: Model trong App thiếu lớp Project/BatchNorm so với file .pth")
            raise e
            
        decoder.eval()
        
        package = { "decoder": decoder, "stoi": vocab["stoi"], "itos": vocab["itos"] }
        self.loaded_models[model_type] = package
        return package

    def predict(self, image_pil, model_choice="CLIP"):
        pkg = self._load_decoder(model_choice)
        decoder = pkg["decoder"]
        stoi = pkg["stoi"]
        itos = pkg["itos"]
        
        features = None
        with torch.no_grad():
            if model_choice == "CLIP":
                img_tensor = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
                features = self.clip_model.encode_image(img_tensor).float()
            else:
                inputs = self.blip_processor(images=image_pil, return_tensors="pt").to(self.device)
                features = self.blip_vision(**inputs).pooler_output
        
        with torch.no_grad():
            # Pass device vào hàm generate
            tokens = decoder.generate(features, stoi, itos, max_len=30, device=self.device)
            
        caption_words = []
        for t in tokens[0]:
            word = itos.get(t.item(), "<unk>")
            if word == "<eos>": break
            if word not in ["<bos>", "<pad>"]:
                caption_words.append(word)
                
        return " ".join(caption_words).replace("_", " ").capitalize()