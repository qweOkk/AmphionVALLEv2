import torch
import random
import glob
import librosa
from utils.g2p.g2p import phonemizer_g2p as g2p
import os
import torchaudio
import re
import numpy as np

test_wer=False
test_sim=False
test_fid=False

class WER:
    def __init__(self):
        print("Loading WER")
        from transformers import Wav2Vec2Processor, HubertForCTC

        from evaluate import load
        wer = load("wer")

        self.wer = wer
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = self.model.to("cuda")

    def calc(self, transcript_text, target_text):
        transcript_text = transcript_text.upper()
        transcript_text = re.sub(r"[^\w\s]", "", transcript_text)
        transcript_text = re.sub(r"\s+", " ", transcript_text)
        transcript_text = transcript_text.strip()

        target_text = target_text.upper()
        target_text = re.sub(r"[^\w\s]", "", target_text)
        target_text = re.sub(r"\s+", " ", target_text)
        target_text = target_text.strip()

        predictions = [transcript_text]
        references = [target_text]
        wer_score = self.wer.compute(predictions=predictions, references=references)
        return wer_score

    def __call__(self, audio, gt_text):
        # need 16khz audio, 1-dimensional
        assert len(audio.shape) == 1
        audio = np.array(audio.cpu())
        input_values = self.processor(audio, return_tensors="pt").input_values.to("cuda")
        logits = self.model(input_values=input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript_text = self.processor.decode(predicted_ids[0])
        # remove special characters
        transcript_text = re.sub(r"[^\w\s]", "", transcript_text)

        wer_score = self.calc(transcript_text, gt_text)
        return wer_score

class SIM:
    def __init__(self):
        from evaluation_test.eval import WAVLM_LARGE_FINTUNED_PATH, load, init_model, pipeline, Tasks

        print("Loading WavLM-large-finetuned")
        self.speaker_encoder = init_model(checkpoint=WAVLM_LARGE_FINTUNED_PATH).to("cuda").eval()
    def __call__(self, audio1, audio2):
        breakpoint()
        audio1 = audio1.unsqueeze(0).to("cuda")
        audio2 = audio2.unsqueeze(0).to("cuda")
        with torch.no_grad():
            embedding1 = self.speaker_encoder(audio1)
            embedding2 = self.speaker_encoder(audio2)
            sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=1)
        return sim.item()

class FID:
    pass

class LibriSpeechDevDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, use_vocos=False):
        self.data_dir = '/mnt/petrelfs/hehaorui/jiaqi/LibriSpeech/test-clean/*/*'
        self.wav_list = glob.glob(self.data_dir + '/*.flac') + glob.glob(self.data_dir + '/*.wav')
        random.shuffle(self.wav_list)

        self.transcript_file = glob.glob(self.data_dir + '/*.txt')
        self.transcripts = {}
        for f_transcript in self.transcript_file:
            with open(f_transcript, 'r') as f:
                for line in f:
                    line = line.strip().split()
                    self.transcripts[line[0]] = ' '.join(line[1:])

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_file = self.wav_list[idx]
        transcript = self.transcripts[os.path.basename(wav_file)[:-5]]
        orig_transcript = transcript
        transcript = g2p(transcript, 'en')[1]
        transcript = torch.tensor(transcript, dtype=torch.long)

        speech, _ = librosa.load(wav_file, sr=16000)
        # resample to 24k
        speech = librosa.resample(speech, orig_sr=16000, target_sr=24000)
        speech = torch.tensor(speech, dtype=torch.float32)

        return {
            'speech': speech,
            'phone_ids': transcript,
            'transcript': orig_transcript,
            'target_transcript': orig_transcript,
            'output_path': os.path.basename(wav_file)[:-5] + '.wav',
        }


def test():
  # initialize a librispeech test dataset instance.
  # This requires having a librispeech test set.
  # Alternatively, you could write your own dataset instance, as long as it returns similar entries.
    dataset = LibriSpeechDevDataset()
    from .valle_inference import ValleInference
    inference = ValleInference(use_vocos=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    import tqdm

    for num_beams in [1]:
            for top_k in [10]:
                for top_p in [1.0]:
                    for repeat_penalty in [1.15]:
                        for temperature in [1.1]:
        
    
                            for batch in tqdm.tqdm(dataloader):
                                if batch['speech'].shape[-1] < 10*24000:
                                    continue
                                print(batch['transcript'][0].lower())
                                chunks = []
                                chunks += [dict(top_p=top_p,
                                    top_k=top_k,
                                    temperature=temperature,
                                    num_beams=num_beams,
                                    repeat_penalty=repeat_penalty,
                                    max_length=2000,)]
                                # chunks += [dict(top_p=top_p,
                                #     top_k=top_k,
                                #     temperature=temperature,
                                #     num_beams=num_beams,
                                #     repeat_penalty=1.05,
                                #     max_length=75*3+75+start_length,)]
                                # chunks += [dict(top_p=top_p,
                                #     top_k=top_k,
                                #     temperature=temperature,
                                #     num_beams=num_beams,
                                #     repeat_penalty=repeat_penalty,
                                #     max_length=75+75+start_length+2000,)]
                                output_wav = inference(batch, chunks)
                                torchaudio.save(f'recovered_audio_ar_{num_beams}_{top_k}_{top_p}_{repeat_penalty}_{temperature}.wav', output_wav[0].cpu(), 24000)
                                print(f'recovered_audio_ar_{num_beams}_{top_k}_{top_p}_{repeat_penalty}_{temperature}.wav')

    
if __name__ == '__main__':
    test()
