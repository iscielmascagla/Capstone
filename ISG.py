import streamlit as st
import os
import pandas as pd
import speech_recognition as sr
import cv2
import numpy as np
import google.generativeai as genai
from transformers import pipeline  # Hugging Face Transformers kütüphanesi
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

generation_config={
    "temperature":1,
    "top_p":0.95,
    "top_k":40,
    "max_output_tokens": 8192,
    "response_mime_type":"text/plain",
}

model=genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,

)





# Uygulama başlığı
st.set_page_config(page_title="Kullanıcı Geri Bildirim Sistemi", layout="wide")

# Dosya yolları
ENV_DATA_FILE = "environment_data.csv"
FEEDBACK_DATA_FILE = "feedback_data.csv"

# ADMIN panel parolası
ADMIN_PASSWORD = "admin123"

# CSV'den verileri yükleme
def load_data(file_path, columns=None):
    if os.path.exists(file_path):
        return pd.read_csv(file_path).to_dict("records")
    return [] if columns is None else [dict(zip(columns, [""] * len(columns)))]


# CSV'ye veri kaydetme
def save_data(file_path, data, columns):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, index=False)

# Çevre& İş Sağlığı ve Güvenliği verileri
environment_data = load_data(ENV_DATA_FILE, columns=["image", "text", "note"])
feedback_data = load_data(FEEDBACK_DATA_FILE, columns=["feedback", "note"])

# Sayfa yenileme fonksiyonu
def refresh_page():
    st.rerun()  # Burada st.rerun() kullanıyoruz

# Duygu analizi için Hugging Face Pipeline kullanımı
sentiment_analyzer = pipeline("sentiment-analysis")

# Türkçe karşılıkları
object_translations = {
    "first aid kit": "ilk yardım kutusu",
    "chemical bin": "kimyasal bidonu",
    "fire exit": "acil çıkış",
    "fire exit route": "acil kaçış rotası",
    "fire alarm button": "yangın alarm butonu",
    "fire extinguisher": "yangın tüpü",
    "electrical panel": "elektrik paneli",
    "smoke dedector": "duman dedektörü",
    "Person": "insan"
}

# Nesnelerin Türkçe isimleri (seçenekler)
object_options = [
    "İlk Yardım Kutusu",
    "Kimyasal Bidon",
    "Acil Çıkış",
    "Acil Kaçış Rotası",
    "Yangın Alarm Butonu",
    "Yangın Tüpü",
    "Elektrik Panosu",
    "Duman Dedektörü",
    "İnsan"
]

# Nesnelerin İngilizce karşılıkları (arka planda kullanılacak etiketler)
object_mapping = {
    "İlk Yardım Kutusu": "first aid kit",
    "Kimyasal Bidon": "chemical bin",
    "Acil Çıkış": "fire exit",
    "Acil Kaçış Rotası": "fire exit route",
    "Yangın Alarm Butonu": "fire alarm button",
    "Yangın Tüpü": "fire extinguisher",
    "Elektrik Panosu": "electrical panel",
    "Duman Dedektörü": "smoke dedector",
    "İnsan": "person"
}

# Şablon eşleme için referans görüntü klasörü
REFERENCE_IMAGE_FOLDER = "reference_images"

# Şablon eşleme için referans görüntüleri yükle
def load_reference_images(folder):
    reference_images = {}
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            label = os.path.splitext(filename)[0]  # Dosya adını etiket olarak kullan
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            reference_images[label] = image
    return reference_images

# Referans görüntüleri yükle
reference_images = load_reference_images(REFERENCE_IMAGE_FOLDER)

# Ana sayfa
st.title("Hoş Geldiniz!")
choice = st.sidebar.selectbox(
    "Bir seçenek seçin:", 
    ["Çevre& İş Sağlığı ve Güvenliği", "Dilek ve Şikayet", "ADMIN","ÇEVİRİ ARACI"]
)

#Function to translate text
if choice=="ÇEVİRİ ARACI":
    st.header("ÇEVİRİ ARACI")

    def translate_text(text,target_language):
     response =model.generate_content(f"Translate the following text to {target_language}: {text}")
     return response.text

    #Kullanıcı metni girişi
    text_to_translate=st.text_area("Çevirilecek metni girin:")

#Hedef dili seçin
    languages={
        'İngilizce':'en',
        'Almanca':'de',
        'Fransızca':'fr',
     'İspanyolca':'es'
    }
    target_language=st.selectbox("Hedef dili seçin:",list(languages.keys()))

    if st.button("Gönder"):
        if text_to_translate and target_language:
            translated_text=translate_text(text_to_translate,languages[target_language])
            st.success(f"Çevrilen Metin: {translated_text}")
        else:
            st.warning("Lütfen metin ve hedef dili seçin!")

# Çevre& İş Sağlığı ve Güvenliği sayfası
if choice == "Çevre& İş Sağlığı ve Güvenliği":
    st.header("Çevre& İş Sağlığı ve Güvenliği")
    
    # Fotoğraf yükleme sırasında nesneleri seçme
    uploaded_file = st.file_uploader("Fotoğraf yükleyin:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        desired_objects = st.multiselect(
            "Tespit edilmesini istediğiniz nesneler:",
            options=object_options,
            default=object_options  # Başlangıçta tüm seçenekler seçili
        )
    else:
        desired_objects = []

    text_input = st.text_area("Metin girin:", placeholder="Görüşlerinizi buraya yazabilirsiniz...")
    
    if st.button("Gönder"):
        if uploaded_file and text_input:
            try:
                # Görüntüyü PIL formatına dönüştür
                image = Image.open(uploaded_file)
                uploaded_image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                
                detected_objects = []

                # Şablon eşleme işlemi
                for label, ref_image in reference_images.items():
                    result = cv2.matchTemplate(uploaded_image_cv, ref_image, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    # Eşik belirleme
                    threshold = 0.8
                    if max_val >= threshold:
                        detected_objects.append(object_translations.get(label, label))

                # Türkçe açıklama oluştur
                note = (
                    f"Tespit Edilen İlgili Nesneler: {', '.join(detected_objects)}"
                    if detected_objects
                    else "Hiçbir ilgili nesne tespit edilmedi."
                )

                new_entry = {"image": uploaded_file.name, "text": text_input, "note": note}
                environment_data.insert(0, new_entry)  # Yeni kayıtları en üstte ekle
                save_data(ENV_DATA_FILE, environment_data, ["image", "text", "note"])

                os.makedirs("uploads", exist_ok=True)
                with open(f"uploads/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success("Veriler başarıyla kaydedildi!")
                refresh_page()
            except Exception as e:
                st.error(f"Görsel işlenirken bir hata oluştu: {str(e)}")
        else:
            st.error("Lütfen hem fotoğraf hem de metin girin!")

# Diğer sayfa içerikleri (Dilek ve Şikayet, ADMIN) aynı kalabilir.

      # Dilek ve Şikayet sayfası
elif choice == "Dilek ve Şikayet":
    st.header("Dilek ve Şikayet")
    text_feedback = st.text_area("Metin girin:", placeholder="Dilek veya şikayetinizi buraya yazabilirsiniz...")
    st.write("Ya da ses kaydı alın:")
    record_button = st.button("🎙️ Ses Kaydı Al")
    
    if record_button:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Konuşmaya başlayabilirsiniz...")
            try:
                audio = recognizer.listen(source, timeout=5)
                st.info("Ses kaydı alınıyor...")
                transcript = recognizer.recognize_google(audio, language="tr-TR")
                
                # Duygu analizi
                sentiment = sentiment_analyzer(transcript)
                sentiment_label = sentiment[0]['label']
                
                # Sonucu kaydet
                feedback_data.insert(0, {"feedback": transcript, "note": sentiment_label})
                save_data(FEEDBACK_DATA_FILE, feedback_data, ["feedback", "note"])
                
                st.success("Ses kaydı başarıyla metne çevrildi!")
                st.write(f"Duygu Durumu: {sentiment_label}")  # Duygu durumu sonucu
                refresh_page()
            except sr.UnknownValueError:
                st.error("Ses anlaşılamadı. Lütfen tekrar deneyin.")
            except sr.RequestError:
                st.error("Google API'ye bağlanılamadı. İnternet bağlantınızı kontrol edin.")

    if st.button("Gönder"):
        if text_feedback:
            feedback_data.insert(0, {"feedback": text_feedback, "note": ""})
            save_data(FEEDBACK_DATA_FILE, feedback_data, ["feedback", "note"])
            st.success("Metin başarıyla kaydedildi!")
            refresh_page()
        else:
            st.error("Lütfen bir metin girin veya ses kaydı alın!")

# ADMIN sayfası
elif choice == "ADMIN":
    st.header("ADMIN Paneli")
    
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if not st.session_state["authenticated"]:
        password = st.text_input("Lütfen parolayı girin:", type="password")
        if st.button("Giriş Yap"):
            if password == ADMIN_PASSWORD:
                st.session_state["authenticated"] = True
                st.success("Giriş başarılı! ADMIN paneline erişebilirsiniz.")
                refresh_page()
            else:
                st.error("Yanlış parola! Tekrar deneyin.")
    else:
        admin_choice = st.radio(
            "Bir kategori seçin:", 
            ["Çevre& İş Sağlığı ve Güvenliği", "Dilek ve Şikayet"]
        )

        if admin_choice == "Çevre& İş Sağlığı ve Güvenliği":
            st.subheader("Çevre& İş Sağlığı ve Güvenliği Kayıtları")
            for index, record in enumerate(environment_data):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                     image_path = f"uploads/{record['image']}"
                     if os.path.exists(image_path):
                        st.image(image_path, caption="Yüklenen Fotoğraf", use_container_width=True)
                     else:
                         st.warning("Fotoğraf yüklenemedi!")
                with col2:
                    st.write(record["text"])
                    st.write(f"Not: {record.get('note', '')}")  # Tespit edilen nesneleri burada göster
                    note_input = st.text_area(f"Not (Kayıt {index+1}):", record.get("note", ""))
                    if st.button(f"Notu Kaydet (Kayıt {index+1})", key=f"save_note_env_{index}"):
                        environment_data[index]["note"] = note_input
                        save_data(ENV_DATA_FILE, environment_data, ["image", "text", "note"])
                        st.success(f"Kayıt {index+1} için not kaydedildi.")
                with col3:
                    if st.button(f"Sil (Kayıt {index+1})", key=f"delete_env_{index}"):
                        environment_data.pop(index)
                        save_data(ENV_DATA_FILE, environment_data, ["image", "text", "note"])
                        st.success(f"Kayıt {index+1} silindi.")
                        refresh_page()

        elif admin_choice == "Dilek ve Şikayet":
            st.subheader("Dilek ve Şikayet Kayıtları")
            for index, feedback in enumerate(feedback_data):
                col1, col2 = st.columns([4, 1])
                with col1:
                    feedback_text = str(feedback.get("feedback", ""))
                    st.write(f"- {feedback_text}")
                    note_input = st.text_area(f"Not (Kayıt {index+1}):", feedback.get("note", ""))
                    if st.button(f"Notu Kaydet (Kayıt {index+1})", key=f"save_note_feedback_{index}"):
                        feedback_data[index]["note"] = note_input
                        save_data(FEEDBACK_DATA_FILE, feedback_data, ["feedback", "note"])
                        st.success(f"Kayıt {index+1} için not kaydedildi.")
                with col2:
                    if st.button(f"Sil (Kayıt {index+1})", key=f"delete_feedback_{index}"):
                        feedback_data.pop(index)
                        save_data(FEEDBACK_DATA_FILE, feedback_data, ["feedback", "note"])
                        st.success(f"Kayıt {index+1} silindi.")
                        refresh_page()
