# SMART-NAR_Fast_TTS
This repository is the official implementation of SMART-Long_Fast_TTS

## Environment
Under Python 3.6

## Requirements
To install requirements:
<pre>
<code>
pip install -r requirements.txt
</code>
</pre>

## Training
To train the NAR TTS model, run this command:
<pre>
<code>
python3 train.py --stage tts --model dcgantts --conf dcgantts_v1.yaml
</code>
</pre>

To train the vocoder, run this command:      
<pre>
<code>
python3 train.py --stage voc --model melgan --conf melgan_v1.yaml
</code>
</pre>

## Evaluation
To evaluate, run:
<pre>
<code>
write your own code here
</code>
</pre>

## Pre-trained Models
You can download pretrained models here:
* <http://example.com/>

## Results
Our model's performance is here:
