from decimal import Decimal, getcontext
import unicodedata, re, time
from bisect import bisect_left

getcontext().prec = 50  # DDC 소수부 비교 정밀도
JAMO_ORDER = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JAMO_IDX = {ch: i for i, ch in enumerate(JAMO_ORDER)}

def _to_initial_jamo(ch):
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:  # 완성형
        return (code - 0xAC00) // 588
    return JAMO_IDX.get(ch, -1)

def _letter_code(ch):
    # 영문(A..Z) < 한글(초성순) < 기타
    if 'A' <= ch <= 'Z':
        return (0, ord(ch))
    j = _to_initial_jamo(ch)
    if j != -1:
        return (1, j)
    return (2, ord(ch))

def _normalize(s):
    s = unicodedata.normalize('NFKC', (s or "").upper())
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace('..', '.')
    s = re.sub(r'(?<=\d)O(?=\d)', '0', s)     # O↔0
    s = re.sub(r'(?<=\d)[IL](?=\d)', '1', s)  # I,l↔1
    s = s.replace(',', '.')                   # 004,3 -> 004.3
    return s

_TOK = re.compile(r'([A-Z]+|[ㄱ-ㅎ가-힣]+|\d+|[.])')
def _tokens(s):
    return [t for t in _TOK.findall(s or "") if t]

YEAR = re.compile(r'(?<!\d)((?:19|20)\d{2})(?!\d)')
VOL  = re.compile(r'(?:V|VOL\.?|권)\s*\.?\s*(\d+)', re.I)
PART = re.compile(r'(?:PT|NO)\s*\.?\s*(\d+)', re.I)
COPY = re.compile(r'(?:C|COPY|C\.)\s*\.?\s*(\d+)', re.I)

class DdcKey:
    # 청구기호 클래스
    def __init__(self, cls, dec, cutters, year, vol, part, copy_):
        self.cls = cls
        self.dec = dec
        self.cutters = cutters    # 리스트/튜플 [(grp, code, num), ...]
        self.year = year
        self.vol = vol
        self.part = part
        self.copy = copy_

    def tup(self):
        # 파이썬 튜플 사전식 비교 활용
        return (self.cls, self.dec, tuple(self.cutters), self.year, self.vol, self.part, self.copy)

def parse_ddc_key(raw):
    """DDC 문자열을 DdcKey로 파싱. 실패 시 None."""
    s = _normalize(raw)
    m = re.match(r'^(\d{3})(?:\.(\d+))?', s)  # 004 / 004.3 / 005.133
    if not m:
        return None
    cls = int(m.group(1))
    dec = Decimal('0.' + m.group(2)) if m.group(2) else Decimal(0)
    rest = s[m.end():].strip()

    year = next((int(y) for y in YEAR.findall(rest)), -1)
    vol  = next((int(v) for v in VOL.findall(rest)), -1)
    part = next((int(p) for p in PART.findall(rest)), -1)
    copy_ = next((int(c) for c in COPY.findall(rest)), -1)

    toks = _tokens(rest)
    cutters = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == '.':
            i += 1
            continue
        if re.fullmatch(r'[A-Z]+|[ㄱ-ㅎ가-힣]+', t):
            ch = t[0]
            grp, code = _letter_code(ch)
            num = -1
            if i + 1 < len(toks) and toks[i + 1].isdigit():
                num = int(toks[i + 1]); i += 1
            cutters.append((grp, code, num))
        i += 1

    return DdcKey(cls, dec, cutters, year, vol, part, copy_)