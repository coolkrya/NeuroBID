# 🕵️ Face Recognition MVP
Система идентификации лиц на основе метрического обучения, предобученной нейросети и быстрого векторного поиска.

## 🚀 Features
- ✅ **Предобученная модель**: MobileFaceNet (128-dim), оптимизирована под Apple Silicon (MPS)
- ✅ **Быстрый поиск**: FAISS `IndexFlatIP` (Inner Product = Cosine Similarity для L2-нормированных векторов)
- ✅ **Детерминизм**: Фиксированные seed'ы, стабильные эмбеддинги при повторных запусках
- ✅ **UI**: Gradio-интерфейс с drag-and-drop, настройкой порога и топ-K
- ✅ **Модульность**: Разделение препроцессинга, модели, поиска и обучения

## 🛠 Установка
```bash
# 1. Клонирование и зависимости
git clone <repo_url>
cd face_id_vkr
pip install -r requirements.txt

# 2. Скачивание весов (~4.7 MB)
mkdir -p models
curl -L -o models/mobilefacenet.pth "https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaE1xVlBENDRjRE9oa1NNSG9kU0g0cmhmYjV1&cid=CEC0E1F8F0542A13&id=CEC0E1F8F0542A13%21836&parId=CEC0E1F8F0542A13%21sea8cc6beffdb43d7976fbc7da445c639&o=OneUp"

# 3. Подготовка данных и индекса
python src/prepare_data.py
python rebuild_index.py --split val