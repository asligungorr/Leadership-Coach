import os
import torch
import whisper

# Load the smallest Whisper model (optimized for CPU performance).
whisper_model = whisper.load_model("tiny")

def extract_transcripts_from_downloads():
    # Find all audio files in the Downloads folder.
    downloads_folder = "./downloads"
    
    # All possible audio file extensions
    audio_extensions = ['.wav', '.webm', '.mp3', '.m4a', '.flac']
    
    # List the files
    audio_files = [
        f for f in os.listdir(downloads_folder) 
        if any(f.endswith(ext) for ext in audio_extensions)
    ]
    
    # If there are no files, show a warning
    if not audio_files:
        print("No audio files found in the Downloads folder.!")
        return
    
    # To save transcripts as a `.txt` file
    with open("transcripts.txt", "w", encoding="utf-8") as txt_file:
        for audio_file in audio_files:
            try:
                # Generate the file path
                audio_path = os.path.join(downloads_folder, audio_file)
                
                # Generate the transcript
                print(f"{audio_file} Extracting the transcript...")
                
                # CPU-optimized transcript extraction
                result = whisper_model.transcribe(
                    audio_path, 
                    fp16=False,  # Run strictly in CPU mode
                    language='tr'  # Turkish language support
                )
                
                # Write the title (file name) and transcript to a .txt file
                txt_file.write(f"VIDEO TITLE: {os.path.splitext(audio_file)[0]}\n")
                txt_file.write("=" * 50 + "\n")
                txt_file.write(result["text"] + "\n\n")
                
                print(f"{audio_file} The transcript has been successfully processed.")
            
            except Exception as e:
                print(f"{audio_file} Error occurred during processing: {e}")
                print(f"Error details: {str(e)}")

# Extract the transcripts
print("Starting the transcript extraction process...")
extract_transcripts_from_downloads()
print("All transcripts have been saved to the `transcripts.txt` file.")
