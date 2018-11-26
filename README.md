# 대마초팀 - 악플 분류기

## FFP & TF_IDF
> FFP 와 TF_IDF 방식을 사용하여 전처리를 하였다.

## Model

* __FFP__
	* SVM

* __TF_IF__
	* KNN
	* Navie Bayes
	* SVM

## Model 평가

* __FFP__ : 연산속도가 너무 느리며(약 3분), 정확도는 50% 정도이다.
* __TF_IDF__: 연산속도가 매우 빠르며, 정확도는 68% 이다.

## Progress
* 현재 TF_IDF방식이 FFP방식 보다 연산 속도와 정확도 측면에서 더 좋을 결과를 얻었다.
* Saver 와 loader을 사용하여 학습된 모델을 save하고 load하는 기능을 추가하였다.

## TODO

> 웹페이지나 안드로이드를 사용하여 악플이나 선플을 입력했을 때 판별할 수 있는 service를 만들면 좋을 것 같다.

