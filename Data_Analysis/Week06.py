# Commented out IPython magic to ensure Python compatibility.
import matplotlib
# %matplotlib inline

import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm  

!apt-get update -qq
!apt-get install fonts-nanum* -qq

font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font_name = fm.FontProperties(fname=font_path, size=10).get_name()
print(font_name)
plt.rc('font', family=font_name)

fm._rebuild()
matplotlib.rcParams['axes.unicode_minus'] = False

"""수강생분의 이름과 학번을 입력해주세요."""

print("강현석", "20187100")

"""구글 드라이브 연결"""

from google.colab import drive
drive.mount('/gdrive')

"""작업폴더 경로 설정"""

workspace_path = "/gdrive/My Drive/한밭대 20187100/4-1/데이터 분석/DA_과제6"  # 과제 파일 업로드한 경로 반영
# 작업폴더 경로 참고: 작업폴더 하위에 data 폴더 생성하여 CSV 파일 읽기/쓰기 수행
# A = os.path.join(workspace_path, 'data/winequality-red.csv')  # workspace_path 이용한 경로 설정 예시
# A = '/gdrive/My Drive/Colab Notebooks/data/winequality-red.csv'  # 절대 경로 설정 예시

"""# 텍스트빈도분석 - 1. 영어단어분석

### 영어 단어 분석에 필요한 패키지 준비
"""

!pip install wordcloud  # 워드클라우드 설치

"""NLTK 패키지 링크: https://github.com/nltk/nltk"""

import nltk
nltk.download('all')  # Natural Language ToolKit 모든 패키지 설치

import os
import pandas as pd
import glob
import re
from functools import reduce

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from collections import Counter

from wordcloud import WordCloud, STOPWORDS

import matplotlib
import matplotlib.pyplot as plt

"""# 1. 데이터 준비

### 1-1. 파일 병합
"""

all_files = glob.glob(os.path.join(workspace_path, 'data/myCabinetExcelData*.xls'))

all_files  # 출력하여 내용 확인

all_files_data = []  # 저장할 리스트

for file in all_files:
    data_frame = pd.read_excel(file)
    all_files_data.append(data_frame)

all_files_data[0]  # 출력하여 내용 확인

all_files_data_concat = pd.concat(all_files_data, axis=0, ignore_index=True)

all_files_data_concat  # 출력하여 내용 확인

all_files_data_concat.to_csv(os.path.join(workspace_path, 'data/riss_bigdata.csv'), encoding='utf-8', index = False)

"""### 1-2. 데이터 전처리 (Pre-processing)"""

# 제목 추출
all_title = all_files_data_concat['제목']

all_title  # 출력하여 내용 확인

stopWords = set(stopwords.words("english"))
lemma = WordNetLemmatizer()

"""Python 정규식 보충자료: https://wikidocs.net/4308"""

words = []

for title in all_title:
    EnWords = re.sub(r"[^a-zA-Z]+", " ", str(title))  # 정규식 이용하여 예외단어 -> 공백처리
    EnWordsToken = word_tokenize(EnWords.lower())
    EnWordsTokenStop = [w for w in EnWordsToken if w not in stopWords]
    EnWordsTokenStopLemma = [lemma.lemmatize(w) for w in EnWordsTokenStop]
    words.append(EnWordsTokenStopLemma)

print(words)  # 출력하여 내용 확인

words2 = list(reduce(lambda x, y: x+y,words))
print(words2)  # 작업 내용 확인

"""# 2. 데이터 탐색

## 2-1. 단어 빈도 탐색
"""

count = Counter(words2)

count  # 출력하여 내용 확인

word_count = dict()

for tag, counts in count.most_common(50):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))

"""#### 검색어로 사용한 big'과 'data' 빈도가 압도적으로 많으므로, 이를 제거한다."""

# 검색어로 사용한 'big'과 'data' 항목 제거 하기
del word_count['big']
del word_count['data']

"""## 2-2 단어 빈도 히스토그램"""

# 히스토그램 표시 옵션
plt.figure(figsize=(12,5))
plt.xlabel("word")
plt.ylabel("count")
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation='85')

plt.show()

"""# 3. 분석 모델 구축 및 결과 시각화

## 3-1. 연도별 데이터 수
"""

all_files_data_concat['doc_count'] = 0
summary_year = all_files_data_concat.groupby('출판일', as_index=False)['doc_count'].count()
summary_year  # 출력하여 내용 확인

plt.figure(figsize=(12,5))
plt.xlabel("year")
plt.ylabel("doc-count")
plt.grid(True)

plt.plot(range(len(summary_year)), summary_year['doc_count'])
plt.xticks(range(len(summary_year)), [text for text in summary_year['출판일']])

plt.show()

"""## 3-2. 워드클라우드"""

stopwords=set(STOPWORDS)
wc=WordCloud(background_color='ivory', stopwords=stopwords, width=800, height=600)
cloud=wc.generate_from_frequencies(word_count)

plt.figure(figsize=(8,8))
plt.imshow(cloud)
plt.axis('off')
plt.show()

"""#### - 워드 클라우드에 나타나는 단어의 위치는 실행 할 때마다 달라진다. ☺"""

cloud.to_file(os.path.join(workspace_path, "data/riss_bigdata_wordCloud.jpg"))

"""# 텍스트빈도분석 - 2. 한글 단어 분석

## 한글 단어 분석을 위한 패키지 준비
"""

!pip install konlpy  # KoNLPy 패키지 설치

import json
import re

from konlpy.tag import Okt

from collections import Counter

from wordcloud import WordCloud

"""# 1. 데이터 준비

### 1-1. 파일 읽기
"""

inputFileName = os.path.join(workspace_path, 'data/etnews.kr_facebook_2016-01-01_2018-08-01_4차 산업혁명')
data = json.loads(open(inputFileName+'.json', 'r', encoding='utf-8').read())
data  # 출력하여 내용 확인

"""### 1-2. 분석할 데이터 추출"""

message = ''

for item in data:
    if 'message' in item.keys():
        message = message + re.sub(r'[^\w]', ' ', item['message']) +''

message  # 출력하여 내용 확인

"""### 1-3. 품사 태깅 : 명사 추출"""

nlp = Okt()
message_N = nlp.nouns(message)
message_N  # 출력하여 내용 확인

"""## 2. 데이터 탐색

### 2-1. 단어 빈도 탐색
"""

count = Counter(message_N)

count  # 출력하여 내용 확인

word_count = dict()

for tag, counts in count.most_common(80):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))

"""### 히스토그램"""

plt.figure(figsize=(12,5))
plt.xlabel('키워드')
plt.ylabel('빈도수')
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation='75')

plt.show()

"""### 워드클라우드"""

wc = WordCloud(font_path, background_color='ivory', width=800, height=600)
cloud=wc.generate_from_frequencies(word_count)

plt.figure(figsize=(8,8))
plt.imshow(cloud)
plt.axis('off')
plt.show()

cloud.to_file(inputFileName + '_cloud.jpg')

"""# 실습 과제"""

inputFileName = os.path.join(workspace_path, 'data/naver_news_example')
data = json.loads(open(inputFileName+'.json', 'r', encoding='utf-8').read())
data  # 출력하여 내용 확인

description = ''

for item in data:
    if 'description' in item.keys():
        description = description + re.sub(r'[^\w]', ' ', item['description']) +''

description  # 출력하여 내용 확인

nlp = Okt()
description_N = nlp.nouns(description)
description_N  # 출력하여 내용 확인

count = Counter(description_N)

count  # 출력하여 내용 확인

word_count = dict()

for tag, counts in count.most_common(80):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))

plt.figure(figsize=(12,5))
plt.xlabel('키워드')
plt.ylabel('빈도수')
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation='75')

plt.show()

wc = WordCloud(font_path, background_color='ivory', width=800, height=600)
cloud=wc.generate_from_frequencies(word_count)

plt.figure(figsize=(8,8))
plt.imshow(cloud)
plt.axis('off')
plt.show()

"""#### data/naver_news_example.json 파일의 description 추출하여 키워드 분석하기

- 키워드 분석 과정 코드, 출력결과 보이기
- 최종 결과물로 아래와 같은 히스토그램, 워드클라우드 생성하기 (워드클라우드 글자 위치, 색상은 매번 달라짐)
- 그 외의 조건은 앞의 한글 문서 분석과 동일함 (상위 80개 키워드 분석 등)
"""