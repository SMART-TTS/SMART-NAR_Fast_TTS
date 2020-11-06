""" from https://github.com/keithito/tacotron """
import re
from utils.text import cleaners
from utils.text.symbols import eng_symbols, kor_symbols
from hparams import hparam

cleaner_names = hparam.text_cleaners

# Mappings from symbol to numeric ID and vice versa:
symbols = ""
_symbol_to_id = {}
_id_to_symbol = {}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

def change_symbol(cleaner_names):
  symbols = ""
  global _symbol_to_id
  global _id_to_symbol
  if cleaner_names == ["english_cleaners"]: symbols = eng_symbols
  if cleaner_names == ["korean_cleaners"]: symbols = kor_symbols

  _symbol_to_id = {s: i for i, s in enumerate(symbols)}
  _id_to_symbol = {i: s for i, s in enumerate(symbols)}

change_symbol(cleaner_names)

def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  change_symbol(cleaner_names)
  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    try:
      print(m)
      if m is None:
        sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
        break
      sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
      sequence += _arpabet_to_sequence(m.group(2))
      text = m.group(3)
    except:
      print(text)
      exit()
  # Append EOS token
  sequence.append(_symbol_to_id['~'])
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'

if __name__ == "__main__":
  # print(text_to_sequence('this is test sentence.? ', ['english_cleaners']))
  # print(text_to_sequence('Chapter one of Jane eyre. This is there librivox recording. All librivox recordings are in the public domain. For more information or to volunteer please visit librivox dot org.', ['english_cleaners']))
  # print(text_to_sequence('Recording by Elisabeth Klett.', ['english_cleaners']))
  # print(text_to_sequence('테스트 문장입니다.? ', ['korean_cleaners']))
  # print(_clean_text('AB테스트 문장입니다.? ', ['korean_cleaners']))
  # print(_clean_text('mp3 파일을 홈페이지에서 다운로드 받으시기 바랍니다.',['korean_cleaners']))
  # print(_clean_text("마가렛 대처의 별명은 '철의 여인'이었다.", ['korean_cleaners']))
  # print(_clean_text("열하나, 열둘, 열셋, 열넷, 열다섯, 열여섯, 열일곱, 열여덟, 열아홉, 스물.", ['korean_cleaners']))
  # print(_symbols_to_sequence(_clean_text("열하나, 열둘, 열셋, 열넷, 열다섯, 열여섯, 열일곱, 열여덟, 열아홉, 스물.", ['korean_cleaners'])))
  # print(_clean_text("아줌마는 결혼한 여자를 뜻한다.", ['korean_cleaners']))
  # print(text_to_sequence("‘아줌마’는 결혼한 여자를 뜻한다. ‘아줌마’는 결혼한 여자를 뜻한다.", ['korean_cleaners']))

  change_symbol(["english_cleaners"])
  print(_symbol_to_id, '\n')
  text_eng1 = 'At that time the agents on the framing boards of the follow-up car were expected to perform such a function.'
  text_eng1 = text_to_sequence(text_eng1.rstrip(), ['english_cleaners'])
  sequence_eng1 = sequence_to_text(text_eng1)
  print(text_eng1)
  print(sequence_eng1, '\n')
  text_eng2 = 'For instance, the lead car always is manned by Secret Service agents familiar with the area and with local law enforcement officials;'
  text_eng2 = text_to_sequence(text_eng2.rstrip(), ['english_cleaners'])
  sequence_eng2 = sequence_to_text(text_eng2)
  print(text_eng2)
  print(sequence_eng2, '\n')

  change_symbol(["korean_cleaners"])
  print(_symbol_to_id, '\n')
  text1 = u'만약'
  text1 = text_to_sequence(text1.rstrip(), ['korean_cleaners'])
  sequence_kor1 = sequence_to_text(text1)
  print(text1)
  print(sequence_kor1, '\n')
  text2 = u'만약 금강산 관광사업에 관심이 없었다면 아예 소를 북한에 보내지도 않았을 것이다'
  text2 = text_to_sequence(text2.rstrip(), ['korean_cleaners'])
  sequence_kor2 = sequence_to_text(text2)
  print(text2)
  print(sequence_kor2, '\n')

  print('1')

