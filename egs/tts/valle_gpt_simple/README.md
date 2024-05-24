# VALLE v2

## Installation requirement 

Set up your environemnt as in Amphion README.

espeak-ng is required to run G2p. To install it, you could refer to: 
https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md

For Linux, it should be `sudo apt-get install espeak-ng`.
For Windows, refer to the above link.

Make sure your transformers version is 4.30.2, other versions are not tested.

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