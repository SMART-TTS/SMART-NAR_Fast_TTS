# SMART-NAR_Fast_TTS
FastSpeech2 기반의 SMART-TTS의 Non-autoregressive TTS 모델입니다. 공개된 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "소량 데이터만을 이용한 고품질 종단형 기반의 딥러닝 다화자 운율 및 감정 복제 기술 개발" 과제의 일환으로 공개된 코드입니다.

SMART-TTS_NAR_Fast_TTS 모델 v2.0.0 은 [FastSpeech2 모델](https://github.com/ming024/FastSpeech2)을 기반으로 alignment를 external duration label 없이 모델링하는 non-autoregressive 구조의 TTS 모델입니다.

FastSpeeche2 모델을 기반으로 하여 아래 부분들을 개선하였습니다.

Done
* Acoustic feature 를 encoding 하는 reference encoder 추가
* Linguistic feature 와 acoustic feature 사이의 alignment를 학습하기 위한 attention module 추가
* Alignment 로부터 duration predictor 학습을 위한 duration label 추출
* Predicted duration 을 기반으로 Gaussian upsampling 적용 

# Environment
Under Python 3.6

# Requirements
To install requirements:
<pre>
<code>
pip install -r requirements.txt
</code>
</pre>

# Preprocessing
To preprocess:
<pre>
<code>
python3 preprocess.py --conf {preprocess configuration file path}

</code>
</pre>

# Training
To train the NAR TTS model, run this command:
<pre>
<code>
python3 train.py -p {preprocess config file path} -m {model condig file path} -t {training config file path}
</code>
</pre>

# Evaluation
To evaluate, run:
<pre>
<code>
python3 synthesize.py --text <text> --restore_step {restore step} -p {preprocess config file path} -m {model condig file path} -t {training config file path}
</code>
</pre>

# Results
Synthesized audio samples can be found in ./output/results

현재 ./output/results 저장된 샘플들은 연구실 보유중인 DB를 사용해 학습한 샘플입니다.

# Reference
* <1> [ming024's FastSpeech 2 implementation](https://github.com/ming024/FastSpeech2)
* <2> [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.

본 프로젝트 관련 개선사항들에 대한 기술문서는 [여기](https://drive.google.com/file/d/1iyLF4qD5Lj2hbwUxQ8470AOyYOasi4_L/view?usp=sharing)를 참고해 주세요.
