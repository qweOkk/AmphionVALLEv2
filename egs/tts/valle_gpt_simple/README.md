# VALLE v2

## Installation requirement 

Set up your environemnt as in Amphion README.

espeak-ng is required to run G2p. To install it, you could refer to: 
https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md

For Linux, it should be `sudo apt-get install espeak-ng`.
For Windows, refer to the above link.

Make sure your transformers version is 4.30.2, for other versions, if you meet the error `TypeError: LlamaDecoderLayer.__init__() missing 1 required positional argument: 'layer_idx'`, you can 
change line 35 (`__init__` function in `LlamaNARDecoderLayer` class, from `super().__init__(config=config)` to `super().__init__(config=config, layer_idx=0)` to resolve the issue.

## The model files
File `valle_ar.py` and `valle_nar.py` in this folder are models files, these files can be run directly via `python -m models.tts.valle_gpt_simple.valle_ar` (or `python -m models.tts.valle_gpt_simple.valle_nar`, and will invoke a test which overfits it to a single example.

## Preparing dataset and dataloader
Write your own dataloader for your dataset. 
You can reference the `__getitem__` method in `models/tts/valle_gpt_simple/mls_dataset.py`
It should return a dict of a 1-dimensional tensor 'speech', which is a 24kHz speech; and a 1-dimensional tensor of 'phone', which is the phoneme sequence of the speech.

As long as your dataset returns this in `__getitem__`, it should work.

## Training AR
```bash
sh ./egs/tts/valle_gpt_simple/train_ar_mls.sh
```
## Training NAR
```bash
sh ./egs/tts/valle_gpt_simple/train_nar_mls.sh
```
