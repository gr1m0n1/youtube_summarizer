# -*- encoding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

from datetime import datetime, timedelta, timezone

import whisper
from pytubefix import YouTube
import ollama

def get_audio(url):
    yt = YouTube(url)
    return yt.streams.filter(only_audio=True)[0].download(filename="tmp.mp4")

def get_transcript(url, model_size, lang, format):

    model = whisper.load_model(model_size)

    if lang == "None":
        lang = None
    
    result = model.transcribe(get_audio(url), fp16=False, language=lang)

    if format == "None":
        return result["text"]
    elif format == ".srt":
        return format_to_srt(result["segments"])

def format_to_srt(segments):
    output = ""
    for i, segment in enumerate(segments):
        output += f"{i + 1}\n"
        output += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        output += f"{segment['text']}\n\n"
    return output

def format_timestamp(t):
    hh = t//3600
    mm = (t - hh*3600)//60
    ss = t - hh*3600 - mm*60
    mi = (t - int(t))*1000
    return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d},{int(mi):03d}"

def summarize(prompt: str, ollama_host:str, ollama_model:str) -> str:
    """Summarizes messages using an Ollama LLM model."""
    if not prompt:
        return "No messages to summarize."

    logger.info(f" - Generating Summary with Ollama (model {ollama_model} on host {ollama_host}) ...")
    ollama_client = ollama.Client(ollama_host)

    start_time = datetime.now()
    response = ollama_client.chat(model=ollama_model, messages=[{"role": "user", "content": prompt}])
    logger.info(f" - Generated Summary Response in {datetime.now() - start_time} time")

    return response.get("message", dict()).get("content", "")

langs = ["None"] + sorted(list(whisper.tokenizer.LANGUAGES.values()))
model_size = list(whisper._MODELS.keys())

url = "https://www.youtube.com/watch?v=rTXMo6_wWW4"
model_size = "tiny"
lang = "es"
format = "None"

transcript = get_transcript(url, model_size, lang, format)

promt = "Sumariza la transcripción que te incluyo a continuación.\nTranscripción:\n" + transcript
print(transcript)

summary = summarize(promt, "localhost:11434", "qwen2.5:7b")
print(summary)
