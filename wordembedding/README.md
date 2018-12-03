# TF-IDF 사용한 데이터 전처리

## Introduce
> twitter-korean-text를 사용하여 형태소를 분류하였다.
> * EX)
> 연애하는 거 꼴보기 싫다.
> ['연애', '하는', '거', '꼴', '보기', '싫다']

> 분리된 형태소 별로 TF-IDF값을 구하여서 값이 큰 값은 데이터에서 제외하였다.
>

## Module
* make_csv() : element가 [word, TF-IDF value, word frequency] 로 이루어진 csv파일을 만든다.
* make_txt(threshold) : 만들어진 TF-IDF.csv의 element를 이용하여 simple.txt 의 데이터 중 TF-IDF값이 높은 word를 threshold만큼 삭제한다.

> train과 test 데이터 모두 같은 threshold로 전처리 해준다.

## Model

* Doc2vec



## TODO

> threshold 에 따른 성능비교.


