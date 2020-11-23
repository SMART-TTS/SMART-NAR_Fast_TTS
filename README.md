# SMART-NAR_Fast_TTS
DC-TTS 기반의 SMART-TTS의 Non-autoregressive TTS 모델입니다.
공개된 코드는 2020년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한
"소량 데이터만을 이용한 고품질 종단형 기반의 딥러닝 다화자 운율 및 감정 복제 기술 개발"
과제의 일환으로 공개된 코드입니다.

SMART-TTS_Single_Emotional 모델은 [DC-TTS 모델](https://github.com/chaiyujin/dctts-pytorch)을 기반으로
adversarial training을 적용한 non-autoregressive 구조의 TTS 모델입니다. 

DC-TTS 모델을 기반으로 하여 아래 부분들을 개선하였습니다.

Done
* Mel-spectrogram 과 text sequence를 입력으로 mel-spectrogrma을 추정하는 teacher model 구현
* Text sequence를 입력으로 target 길이를 추정하는 target length predictor 구현
* Text sequence를 입력으로 mel-spectrogram을 추정하는 student model 구현
* Waveform generation을 위한 MelGAN 보코더 통합
* Attention error을 줄이기 위한 attention masking 적용 

To do
* Hyperparameter tuning
* Multi-speaker TTS

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

## Results
Synthesized audio samples can be found in ./decode

현재 ./decode 저장된 샘플들은 연구실 보유중인 DB를 사용해 학습한 샘플이며,
내년초 새로운 한국어 DB 공개 예정에 있습니다.

## Reference
* [1] https://github.com/descriptinc/melgan-neurips
* [2] https://github.com/chaiyujin/dctts-pytorch
