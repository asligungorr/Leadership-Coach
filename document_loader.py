import os
import torch
import whisper

# En küçük Whisper modelini yükle (CPU performansı için optimize)
whisper_model = whisper.load_model("tiny")

def extract_transcripts_from_downloads():
    # İndirme klasöründeki tüm ses dosyalarını bul
    downloads_folder = "./downloads"
    
    # Tüm olası ses dosyası uzantıları
    audio_extensions = ['.wav', '.webm', '.mp3', '.m4a', '.flac']
    
    # Dosyaları listele
    audio_files = [
        f for f in os.listdir(downloads_folder) 
        if any(f.endswith(ext) for ext in audio_extensions)
    ]
    
    # Eğer hiç dosya yoksa uyar
    if not audio_files:
        print("İndirme klasöründe ses dosyası bulunamadı!")
        return
    
    # Transkriptleri txt dosyasına kaydetmek için
    with open("leadership_coach_transcripts.txt", "w", encoding="utf-8") as txt_file:
        for audio_file in audio_files:
            try:
                # Dosya yolunu oluştur
                audio_path = os.path.join(downloads_folder, audio_file)
                
                # Transkript oluştur
                print(f"{audio_file} transkripti çıkarılıyor...")
                
                # CPU için optimize edilmiş transkript çıkarma
                result = whisper_model.transcribe(
                    audio_path, 
                    fp16=False,  # Kesinlikle CPU modunda çalış
                    language='tr'  # Türkçe dil desteği
                )
                
                # Başlık (dosya adı) ve transkripti txt dosyasına yaz
                txt_file.write(f"VIDEO BAŞLIĞI: {os.path.splitext(audio_file)[0]}\n")
                txt_file.write("=" * 50 + "\n")
                txt_file.write(result["text"] + "\n\n")
                
                print(f"{audio_file} transkripti başarıyla işlendi.")
            
            except Exception as e:
                print(f"{audio_file} işlenirken hata: {e}")
                print(f"Hata detayı: {str(e)}")

# Transkriptleri çıkar
print("Transkript çıkarma işlemi başlatılıyor...")
extract_transcripts_from_downloads()
print("Tüm transkriptler leadership_coach_transcripts.txt dosyasına kaydedildi.")
