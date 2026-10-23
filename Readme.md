# RAG Based AI Teaching Model

A RAG (Retrieval-Augmented Generation) pipeline that turns your video course into an AI teaching assistant. Ask questions and it will tell you exactly which video and timestamp covers the topic.

## Prerequisites

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) installed and added to PATH
- [Ollama](https://ollama.com) running locally with these models pulled:
  ```bash
  ollama pull bge-m3
  ollama pull llama3.2
  ```

## Setup

```bash
git clone https://github.com/SaumitraBata/RAG-based-AI-teaching-model.git
cd RAG-based-AI-teaching-model
pip install -r requirements.txt
```

## How to Use on Your Own Data

### Step 1 — Collect your videos
Move all your video files to the `videos/` folder

### Step 2 — Convert to mp3
```bash
python video_to_mp3.py
```

### Step 3 — Convert mp3 to json
```bash
python mp3_to_jsons.py
```
> This uses OpenAI Whisper to transcribe and translate audio to English

### Step 4 — Generate Embeddings
```bash
python preprocess_json.py
```
> This creates `embeddings.joblib` using the `bge-m3` model via Ollama

### Step 5 — Ask Questions
```bash
python process_incoming.py
```
> Type your question and the AI will tell you which video and timestamp covers it

## Project Structure

```
├── videos/              # Put your video files here
├── audios/              # Auto-generated mp3 files
├── jsons/               # Auto-generated transcription jsons
├── video_to_mp3.py      # Step 2: Convert videos to mp3
├── mp3_to_jsons.py      # Step 3: Transcribe mp3 to json
├── preprocess_json.py   # Step 4: Generate embeddings
├── process_incoming.py  # Step 5: Query the AI
└── requirements.txt
```

## Notes

- `audios/`, `jsons/`, and `embeddings.joblib` are excluded from git — you generate them locally
- The pipeline uses [Ollama](https://ollama.com) for both embeddings (`bge-m3`) and LLM inference (`llama3.2`) — everything runs locally, no API key needed
