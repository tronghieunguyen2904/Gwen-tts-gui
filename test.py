import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Load model
model = Qwen3TTSModel.from_pretrained(
    "g-group-ai-lab/gwen-tts-0.6B",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# Recommended generation config for Gwen-TTS
generation_config = dict(
    temperature=0.3,
    top_k=20,
    top_p=0.9,
    max_new_tokens=4096,
    repetition_penalty=2.0,
    subtalker_do_sample=True,
    subtalker_temperature=0.1,
    subtalker_top_k=20,
    subtalker_top_p=1.0,
)

# Generate speech with voice cloning
wavs, sr = model.generate_voice_clone(
text="Dạ chào anh, em gọi trực tiếp từ bộ phận phê duyệt hồ sơ thẻ tín dụng. Hiện tại hồ sơ của mình đang nằm trong danh sách ưu tiên nhận hạn mức lên đến hai trăm triệu đồng, kèm ưu đãi miễn phí thường niên trọn đời. Không biết mình có tiện trao đổi hai phút để em hướng dẫn nhận thẻ ngay không ạ?",
    language="vietnamese",
    ref_audio="voice.wav",
    ref_text="Thân hình lực lưỡng của Harpegnathos venator khiến kiến lửa trông như mấy chấm li ti cạnh chúng.",
)

sf.write("output.wav", wavs[0], sr)
