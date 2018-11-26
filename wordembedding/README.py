# TF-IDF 사용한 악플 분류기

## Twitter-Korean-Text
> twitter-korean-text를 사용하여 형태소를 분류하였다.
> * EX)
> 연애하는 거 꼴보기 싫다.
> ['연애', '하는', '거', '꼴', '보기', '싫다']

> 분리된 형태소들에게 가중치를 준다.
>  * EX)
>  {"연애": 0.8, "하는" : 0.7", "거" : 0.4 , "보기" : 0.7 , "싫다" : 0.9}

> one-hot-encoding 과는 다르게 빈도수가 높은 단어들에게는 가중치를 낮추는 방법을 채택
> => TF-IDF
> 

## Model

* __Navie Bayes__
* __SVM__
* __KNN__

> 세가지 결과 현재 Navie Bayes가 성능이 제일 좋았다.

## TODO

> cross validation을 사용하여 하이퍼파라미터를 찾아 테스트해봐야할 것 같다.
> doc2vec 를 이용하여 위에 Model과의 차이를 확인해봐야할 것 같다.
