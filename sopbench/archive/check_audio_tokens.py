"""Verify audio-off video has fewer tokens than audio-on."""
import os, time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv('.env')
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

orig = 'videos/captaincook4d_samples/12_51_tomatomozzarellasalad.mp4'
noaudio = 'videos/captaincook4d_samples_noaudio/12_51_tomatomozzarellasalad.mp4'

for label, path in [('Audio ON ', orig), ('Audio OFF', noaudio)]:
    vf = client.files.upload(file=path, config=types.UploadFileConfig(mime_type='video/mp4'))
    while vf.state == 'PROCESSING':
        time.sleep(2)
        vf = client.files.get(name=vf.name)
    tok = client.models.count_tokens(
        model='gemini-2.5-flash',
        contents=[types.Part.from_uri(file_uri=vf.uri, mime_type='video/mp4')]
    )
    print(f'{label}: {tok.total_tokens:,} tokens')
    client.files.delete(name=vf.name)
