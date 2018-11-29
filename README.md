# Automatic-Stock-Trading
In Wall Street, the global financial center, the proportion of investment in artificial intelligence has increased gradually since the 2008 financial crisis (Source: Bloomberg)
In addition, artificial intelligence can make reasonable decisions because it does not pay for the inefficiency of investment sentiment in determining whether to invest and scale.(E.g, The one AI system at the time of the global financial crisis in 2008 recorded a 681% )

## Dataset 
We will use each of historical stock price dataset from the Yahoo Finace. The dataset contains the raw time-series data(Open, High, Low, Close*, Adjusted Close*, and Volume).

*Close price adjusted for splits.
** Adjusted close price adjusted for both dividends and splits.

+Yahoo Finance
https://finance.yahoo.com/

## Details of Dataset and Models 
+There are 10 stocks dataset(From 2012 ~ From 2017) and KOSPI index.
+We use 10 stocks and the index not only Jan.2016~Dec.2016 as train/Validation dateset, also use Jan.2017~Dec.2017 as test dateset.
+We use XGBoost model.
-Official website: https://xgboost.readthedocs.io/en/latest/index.html
+XGBoost and other ensemble models is one of learning methods to predict stock prices. Afterwards based on past market data, stocks (listed in KOSPI market) are subject to post-verification(back-testing) and real-time simulation investment.
+We calculate rates of each stock of returns every at the end of each week.

## Requirements 
+ XGboost (0.7)
+ numpy (1.15.1)
+ matplotlib (2.2.2)
+ scikit-learn (0.19.1)
+ Pandas (0.22.0)
+ Scipy (1.1.0)

## License
[Apache License 2.0](https://github.com/OpenXAIProject/tutorials/blob/master/LICENSE "Apache")

## Contacts
If you have any question, please contact Eunji Bang(eunji@unist.ac.kr).

<br /> 
<br />

# XAI Project 

**This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2017-0-01779, A machine learning and statistical inference framework for explainable artificial intelligence)**

+ Project Name : A machine learning and statistical inference framework for explainable artificial intelligence(의사결정 이유를 설명할 수 있는 인간 수준의 학습·추론 프레임워크 개발)

+ Managed by Ministry of Science and ICT/XAIC <img align="right" src="http://xai.unist.ac.kr/static/img/logos/XAIC_logo.png" width=300px>

+ Participated Affiliation : UNIST, Korea Univ., Yonsei Univ., KAIST, AItrics  

+ Web Site : <http://openXai.org>
