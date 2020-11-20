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

## Preprocessing
To preprocess:
<pre>
<code>
python3 preprocess.py --conf {configuration file path}

e.g. > python3 preprocess.py --conf model/tts/dcgantts/conf/dcgantts_v1.yaml
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
python3 inference.py --conf decode_v1.yaml
</code>
</pre>

## Reference
* [1] https://github.com/descriptinc/melgan-neurips
* [2] https://github.com/chaiyujin/dctts-pytorch
