import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator

# Загрузка модели
model = load_model('text_classification_model.h5')

# Загрузка токенизатора
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

VOCAB_SIZE = 50000        # максимальное количество слов, которые будут использоваться
SEQUENCE_LENGTH = 400     # максимальное количество слов в каждой последовательности

def classify_text(Text):
    # Перевод текста с русского на английский
    translator = GoogleTranslator(source='ru', target='en')
    translated_text = translator.translate(Text)

    # Преобразование текста в последовательность индексов
    sequences = tokenizer.texts_to_sequences([translated_text])
    padded_sequences = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)

    # Предсказание
    predicted_prob = model.predict(padded_sequences)
    predicted_class = (predicted_prob > 0.95).astype(int)

    if predicted_class[0][0] == 0:
        return 'Информационный'
    else:
        return 'Рекламный'


# Создание интерфейса Gradio
iface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=2, placeholder="Введите текст для классификации..."),
    outputs="text",
    title="Текстовая классификация",
    description="Классифицируйте текст как рекламный или информационный."
)

# Запуск интерфейса
iface.launch()
