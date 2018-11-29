# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:21:29 2018

@author: UNIST
"""
import eikon as ek
import datetime
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from matplotlib import pyplot
from xgboost import plot_importance
import matplotlib.pyplot as plt
from scipy import stats

#######################################load historical price from Eikon 
#kospi_his= ek.get_timeseries('.KS11' , fields=['CLOSE','HIGH', 'LOW', 'OPEN', 'VOLUME'], start_date='2015-1-1', end_date='2018-11-27')
#samsungelectronics= ek.get_timeseries('005930.KS' , fields=['CLOSE','HIGH', 'LOW', 'OPEN', 'VOLUME'], start_date='2015-1-1', end_date='2018-11-27')
#skhynix= ek.get_timeseries('000660.KS' , fields=['CLOSE','HIGH', 'LOW', 'OPEN', 'VOLUME'], start_date='2015-1-1', end_date='2018-11-27')
#hyundaimotor= ek.get_timeseries('005380.KS' , fields=['CLOSE','HIGH', 'LOW', 'OPEN', 'VOLUME'], start_date='2015-1-1', end_date='2018-11-27')
#lgchem= ek.get_timeseries('051910.KS' , fields=['CLOSE','HIGH', 'LOW', 'OPEN', 'VOLUME'], start_date='2015-1-1', end_date='2018-11-27')
#hyundaimobis= ek.get_timeseries('012330.KS' , fields=['CLOSE','HIGH', 'LOW', 'OPEN', 'VOLUME'], start_date='2015-1-1', end_date='2018-11-27')
#samsungcnt= ek.get_timeseries('028260.KS' , fields=['CLOSE','HIGH', 'LOW', 'OPEN', 'VOLUME'], start_date='2015-1-1', end_date='2018-11-27')
#koreaelectric= ek.get_timeseries('015760.KS' , fields=['CLOSE','HIGH', 'LOW', 'OPEN', 'VOLUME'], start_date='2015-1-1', end_date='2018-11-27')
#samsungsds= ek.get_timeseries('018260.KS' , fields=['CLOSE','HIGH', 'LOW', 'OPEN', 'VOLUME'], start_date='2015-1-1', end_date='2018-11-27')
#amorepacific= ek.get_timeseries('090430.KS' , fields=['CLOSE','HIGH', 'LOW', 'OPEN', 'VOLUME'], start_date='2015-1-1', end_date='2018-11-27')
#lghousehold= ek.get_timeseries('051900.KS' , fields=['CLOSE','HIGH', 'LOW', 'OPEN', 'VOLUME'], start_date='2015-1-1', end_date='2018-11-27')

########################################### save downloaded stock data
#pd.DataFrame.to_csv(kospi_his,'kospi.csv')
#pd.DataFrame.to_csv(samsungelectronics,'samsungelec.csv')
#pd.DataFrame.to_csv(skhynix,'skhynix.csv')
#pd.DataFrame.to_csv(hyundaimotor,'motors.csv')
#pd.DataFrame.to_csv(lgchem,'lgchem.csv')
#pd.DataFrame.to_csv(samsungcnt,'samsungcnt.csv')
#pd.DataFrame.to_csv(hyundaimobis,'mobis.csv')
#pd.DataFrame.to_csv(koreaelectric,'koreaelec.csv')
#pd.DataFrame.to_csv(samsungsds,'samsungsds.csv')
#pd.DataFrame.to_csv(amorepacific,'amorepacific.csv')
#pd.DataFrame.to_csv(lghousehold,'lghousehold.csv')

#######################################call saved dataset
#path='Put your path here'
#kospi=pd.read_csv(os.path.join(path,'kospi.csv'))
#amorepacific=pd.read_csv(os.path.join(path,'amorepacific.csv'))
#samsungelectronics=pd.read_csv(os.path.join(path,'samsungelec.csv'))
#skhynix=pd.read_csv(os.path.join(path,'skhynix.csv'))
#hyundaimotor=pd.read_csv(os.path.join(path,'motors.csv'))
#lgchem=pd.read_csv(os.path.join(path,'lgchem.csv'))
#samsungcnt=pd.read_csv(os.path.join(path,'samsungcnt.csv'))
#hyundaimobis=pd.read_csv(os.path.join(path,'mobis.csv'))
#koreaelectric=pd.read_csv(os.path.join(path,'koreaelec.csv'))
#samsungsds=pd.read_csv(os.path.join(path,'samsungsds.csv'))
#lghousehold=pd.read_csv(os.path.join(path,'lghousehold.csv'))


def stock_predict(zz,name):
    path='user/put/your/path/here'
    kospi=pd.read_csv(os.path.join(path,'kospi.csv'))
    amorepacific=pd.read_csv(os.path.join(path,'amorepacific.csv'))
    samsungelectronics=pd.read_csv(os.path.join(path,'samsungelec.csv'))
    skhynix=pd.read_csv(os.path.join(path,'skhynix.csv'))
    hyundaimotor=pd.read_csv(os.path.join(path,'motors.csv'))
    lgchem=pd.read_csv(os.path.join(path,'lgchem.csv'))
    samsungcnt=pd.read_csv(os.path.join(path,'samsungcnt.csv'))
    hyundaimobis=pd.read_csv(os.path.join(path,'mobis.csv'))
    koreaelectric=pd.read_csv(os.path.join(path,'koreaelec.csv'))
    samsungsds=pd.read_csv(os.path.join(path,'samsungsds.csv'))
    lghousehold=pd.read_csv(os.path.join(path,'lghousehold.csv'))
    
    
    aaa=zz.iloc[246:489,1]  #2017
    #aaa=zz.iloc[248:494,1]   #2016
    aaa=np.array(aaa)
    
    #stock_list_name=[kospi,'amorepacific','samsungelectronics','skhynix','hyundaimotor','lgchem','samsungcnt',
    #                 'hyundaimobis','koreaelectric','samsungsds','lghousehold']
    
    ######################date list per week from the beginning of year to the end of year(In this case, 2017)
    date_start=[]
    date_end=[]
    date_list_week_start=[]
    date_list_week_end=[]
    for i in range(52):
        date_s=datetime.datetime(2017,1,2)
        date_=datetime.datetime(2017,1,6)
#        date_s=datetime.datetime(2016,1,4)
#        date_=datetime.datetime(2016,1,8)
        date_list_week_start=date_s + datetime.timedelta(7)*i
        date_list_week_end=date_+datetime.timedelta(7)*i
        date_start.append(date_list_week_start)
        date_end.append(date_list_week_end)
    
    #######Use date_start and date_end
    ############################################dates for train data
    #date_2016_start=datetime.datetime(2017,8,4)
    #data_2016_end=date_s+datetime.timedelta(-3)
    start_2016_date=[]
    end_2016_date=[]
    for i in range(52):
        date_2016_start=datetime.datetime(2016,7,4)
        #date_2016_start=datetime.datetime(2015,1,4)
        data_2016_end=date_s+datetime.timedelta(-3)
        week_2016_start=date_2016_start + datetime.timedelta(7)*i
        week_2016_end=data_2016_end+datetime.timedelta(7)*i
        start_2016_date.append(week_2016_start)
        end_2016_date.append(week_2016_end)
    ##### USe start_2016_date & end_2016_date    
    
    train_end_date=[]
    for i in range(52):
        train_end_date.append(end_2016_date[i]-datetime.timedelta(7))
    
    #########################################################train/validation dataset
    
    date_info=koreaelectric.iloc[:,0]
    given_date=pd.DataFrame(date_info,columns=['Date'])

    #########################################################train/validation dataset

    #######################################################################scaling

    amorepacific=zz.copy()
    #amorepacific=stats.zscore(amorepacific.iloc[:,1:])
#    amorepacific=stats.zscore(amorepacific.iloc[:,1])
    #amorepacific = scaler.fit_transform(amorepacific.iloc[:,1:])
    #kospi = scaler.transform(kospi.iloc[:,1:])
    
    amorepacific = amorepacific.iloc[:,1]
    amorepacific=pd.DataFrame(amorepacific,columns=['CLOSE'])
    #amorepacific=pd.DataFrame(amorepacific,columns=['CLOSE','HIGH','LOW','OPEN','VOLUME'])
    amorepacific=pd.concat([amorepacific,given_date],axis=1)
    
    kospi = kospi.iloc[:,1]
    kospi=pd.DataFrame(kospi,columns=['CLOSE'])
    #kospi=pd.DataFrame(kospi,columns=['CLOSE','HIGH','LOW','OPEN','VOLUME'])
    kospi=pd.concat([kospi,given_date],axis=1)
   

    ###########################################################################

    amore = amorepacific.copy()
    amore['Date'] = amore['Date'].apply((lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')))
    kospi['Date'] = kospi['Date'].apply((lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')))
    
    
    kospi_list=[]
    amore_list = []
    x_train=[]
    y_train_=[]
    x_test=[]
    y_test_=[]
    xgb_prediction_total=[]
    x_train=[]
    y_train_=[]
    x_test=[]
    y_test_=[]
    
    
    
    for s,e in zip(start_2016_date, end_2016_date):
        amore_list.append(amore.loc[(amore['Date']>=s) & (amore['Date']<=e)])
        kospi_list.append(kospi.loc[(kospi['Date']>=s) & (kospi['Date']<=e)])
    
    for i in range(52):#len(kospi_list)):
        amore_gen_fea=generate_features1(amore_list[i])
        kospi_gen_fea=generate_features1(kospi_list[i])
        kospi_gen_fea.rename(columns={c:"kospi_"+c for c in kospi_gen_fea.columns}, inplace=True)
        amore_gen_results=pd.concat([amore_gen_fea, kospi_gen_fea], axis=1, join_axes=[amore_gen_fea.index])
        amore_gen_results['Date']=kospi['Date']
        for i in range(5):
            amore_gen_results['CLOSE-'+str(i+1)] = amore_gen_results['CLOSE'].shift((i+1))
        pred_date = 1
        for i in range(pred_date):
            amore_gen_results['CLOSE+'+str(i+1)] = amore_gen_results['CLOSE'].shift(-1*(i+1))
        amore_gen_results.dropna(inplace=True)
        train_idx = amore_gen_results.index[:int(0.8*len(amore_gen_results))]
        result_subset=amore_gen_results.loc[train_idx]
        y_col_amore = ['CLOSE+'+str(i+1) for i in range(pred_date)]
        X_col_amore = amore_gen_results.drop(['Date']+y_col_amore,axis=1).columns
        X_train = result_subset[X_col_amore]
        y_train = result_subset[y_col_amore]
        train_dates = result_subset['Date']
        valid_idx = amore_gen_results.index[int(0.8*len(amore_gen_results)):]
        validation_subset1=amore_gen_results.loc[valid_idx]
        X_test=validation_subset1[X_col_amore]
        y_test=validation_subset1[y_col_amore]
        test_dates = validation_subset1['Date']
        ##############################################################################build model

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_train = X_scaler.fit_transform(X_train)
        y_train = y_scaler.fit_transform(y_train)
        X_test = X_scaler.transform(X_test)
        y_test = y_scaler.transform(y_test)
        
        ##############################################
        #############Put your model here#############
        #############################################
        
        print("Best rmse: {:.2f} with {} rounds".format(model.best_score,model.best_iteration+1))
        
    
        train_pred=model.predict(dtrain)
        test_pred=model.predict(dtest)
        plt.figure(figsize=(11,5))
        plt.plot(train_dates, y_train, label='train y')
        plt.plot(train_dates, train_pred, label='train pred')
        plt.plot(test_dates, y_test, label='test y')
        plt.plot(test_dates, test_pred, label='test pred')
        #plt.legend(True)
        plt.show()
        xgb_prediction_total.append(test_pred)


    ###################################################################################
    
    total_prediction=pd.DataFrame(xgb_prediction_total)
    total_prediction_trans=total_prediction.T
    
    
    command=[]
    total_prediction_trans = total_prediction_trans.fillna(10000)
    
    #pd.DataFrame.to_csv(total_prediction_trans,'total_prediction_v1.csv')

        ###########################################################
        #############Put your trading strategies here#############
        ########################################################
   
    ######################################################

    cal_list_date=pd.read_csv(os.path.join(path,'list.csv'))
    cal_list_date=np.array(cal_list_date)
    gg = []
    ind = 0
    index = []
    index_last = []
    hh = []
    ggg = 0
    
    
    ##############################################calculate your return
    ###############Weekly
    if cal_list_date[i] != 0:
        if command[ggg] == 'sell stock':
            gg.append(np.log(aaa[ind]/aaa[int(ind+cal_list_date[i]-1)]))
        elif command[ggg] == 'buy stock':
            gg.append(np.log(aaa[int(ind+cal_list_date[i]-1)]/aaa[ind]))
        else:
            gg.append(0)
        hh.append(cal_list_date[i])
        index.append(ind)
        index_last.append(ind+cal_list_date[i]-1)
        ind = ind + cal_list_date[i]
        ggg = ggg + 1
    

   

    #########Annualy
    gg_aaa=[]
    prod_append=[]
    for i in range(len(gg)):
        gg_first=(1+gg[i])
        gg_aaa.append(gg_first)
        aa_prod=np.product(gg_aaa)-1 

    print(name+' Annual total return: '+str(aa_prod))
    prod_append.append(name)
    prod_append.append(aa_prod)

    return prod_append

 
####Run the function.
acc_list = []
acc_list.append(stock_predict(amorepacific,'amorepacific'))
acc_list.append(stock_predict(samsungelectronics,'samsungelectronics'))
acc_list.append(stock_predict(skhynix,'skhynix'))
acc_list.append(stock_predict(hyundaimotor,'hyundaimotor'))
acc_list.append(stock_predict(lgchem,'lgchem'))
acc_list.append(stock_predict(samsungcnt,'samsungcnt'))
acc_list.append(stock_predict(hyundaimobis,'hyundaimobis'))
acc_list.append(stock_predict(koreaelectric,'koreaelectric'))
acc_list.append(stock_predict(samsungsds,'samsungsds'))
acc_list.append(stock_predict(lghousehold,'lghousehold'))


def generate_features1(df):
    df_new=pd.DataFrame()
    df_new['CLOSE']=df['CLOSE']
    df_new['close_1']=df['CLOSE']-df['CLOSE'].shift(1)
    df_new['close_7']=df['CLOSE']-df['CLOSE'].shift(7)
    fill_mean = np.mean(df_new['close_7'][:])
    df_new = df_new.fillna(value=fill_mean)
    
    return df_new
