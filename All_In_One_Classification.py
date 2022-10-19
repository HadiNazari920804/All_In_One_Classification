def All_In_One_Classification(x,y):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn import tree
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    
    ac=range(100)
    ts=0.1
    
    #LDA
    
    l=[]
    for i in ac:
        X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=ts)
        lda=LinearDiscriminantAnalysis()
        lda.fit(X_train,Y_train)
        l.append(lda.score(X_test,Y_test))
    LDA_Ac_mean=np.mean(l)
    LDA_Ac_std=np.std(l)
    
    
    
    #QDA
    q=[]
    for i in ac:
        X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=ts)
        qda=LinearDiscriminantAnalysis()
        qda.fit(X_train,Y_train)
        q.append(qda.score(X_test,Y_test))
    QDA_Ac_mean=np.mean(q)
    QDA_Ac_std=np.std(q)
    
    
    #Naive Bayes
    g=[]
    for i in ac:
        X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=ts)
        gnb=GaussianNB()
        gnb.fit(X_train,Y_train)
        g.append(gnb.score(X_test,Y_test))
    NaiveBayes_Ac_mean=np.mean(g)
    NaiveBayes_Ac_std=np.std(g)
    
    
    #Logestic Regression
    lore=[]
    for i in ac:
        X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=ts)
        lr=LogisticRegression(penalty='none',solver='newton-cg')
        lr.fit(X_train,Y_train)
        lore.append(lr.score(X_test,Y_test))
    LogesticRegression_Ac_mean=np.mean(lore)
    LogesticRegression_Ac_std=np.std(lore)
    
    
    #KNN
    
    nneighbors=np.arange(1,int(np.sqrt(X_train.shape[0]))+1)
    CVscores=np.empty((len(nneighbors),2))
    CV=10
    counter=-1
    
    for k in nneighbors:
        counter+=1
        knn=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
        CVscores[counter,:]=np.array([k,np.mean(cross_val_score(knn,X_train,Y_train,cv=CV))])
    BestK=int(CVscores[np.argmax(CVscores[:,1]),:][0])
    
    k=[]
    for i in ac:
        X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=ts)
        knno=KNeighborsClassifier(n_neighbors=BestK,metric='minkowski',p=2)
        knno.fit(X_train,Y_train)
        k.append(knno.score(X_test,Y_test))
    KNN_Ac_mean=np.mean(k)
    KNN_Ac_std=np.std(k)
    
    #Decision Tree
    md=np.arange(1,501)
    CVscores=np.empty((len(md),2))
    CV=10
    counter=-1
    
    for t in md:
        counter+=1
        dectree=tree.DecisionTreeClassifier(max_depth=t)
        CVscores[counter,:]=np.array([t,np.mean(cross_val_score(dectree,X_train,Y_train,cv=CV))])
    Bestmd=int(CVscores[np.argmax(CVscores[:,1]),:][0])
    
    dt=[]
    for i in ac:
        X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=ts)
        dectree=tree.DecisionTreeClassifier(max_depth=Bestmd)
        dectree.fit(X_train,Y_train)
        dt.append(dectree.score(X_test,Y_test))
    DecisionTree_Ac_mean=np.mean(dt)
    DecisionTree_Ac_std=np.std(dt)
    
    
    #bagging
    nestimator=np.arange(1,101)
    CVscores=np.empty((len(nestimator),2))
    CV=10
    counter=-1
    
    for b in nestimator:
        counter+=1
        bag=BaggingClassifier(n_estimators=b)
        CVscores[counter,:]=np.array([b,np.mean(cross_val_score(bag,X_train,Y_train,cv=CV))])
    Bestnestimator=int(CVscores[np.argmax(CVscores[:,1]),:][0])
    
    bg=[]
    for i in ac:
        X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=ts)
        bag=BaggingClassifier(n_estimators=Bestnestimator)
        bag.fit(X_train,Y_train)
        bg.append(bag.score(X_test,Y_test))
    Bagging_Ac_mean=np.mean(bg)
    Bagging_Ac_std=np.std(bg)
    
    
    #Random Forrest
    nestimatorrf=np.arange(1,1001,100)
    CVscores=np.empty((len(nestimatorrf),2))
    CV=10
    counter=-1
    
    for r in nestimatorrf:
        counter+=1
        rf=RandomForestClassifier(n_estimators=r,max_features='auto',max_depth=Bestmd)
        CVscores[counter,:]=np.array([r,np.mean(cross_val_score(rf,X_train,Y_train,cv=CV))])
    Bestnestimatorrf=int(CVscores[np.argmax(CVscores[:,1]),:][0])
    
    
    rnfo=[]
    for i in ac:
        X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=ts)
        rf=RandomForestClassifier(n_estimators=Bestnestimatorrf,max_features='auto',max_depth=Bestmd)
        rf.fit(X_train,Y_train)
        rnfo.append(rf.score(X_test,Y_test))
    RandomForrest_Ac_mean=np.mean(rnfo)
    RandomForrest_Ac_std=np.std(rnfo)
    
    
    #Boosting
    nestimatorboost=np.arange(1,2001,100)
    CVscores=np.empty((len(nestimatorboost),2))
    CV=10
    counter=-1
    
    for bo in nestimatorboost:
        counter+=1
        boost=AdaBoostClassifier(n_estimators=bo)
        CVscores[counter,:]=np.array([bo,np.mean(cross_val_score(boost,X_train,Y_train,cv=CV))])
    Bestnestimatorboost=int(CVscores[np.argmax(CVscores[:,1]),:][0])
    
    boos=[]
    for i in ac:
        X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=ts)
        boost=AdaBoostClassifier(n_estimators=Bestnestimatorboost)
        boost.fit(X_train,Y_train)
        boos.append(boost.score(X_test,Y_test))
    Boosting_Ac_mean=np.mean(boos)
    Boosting_Ac_std=np.std(boos)
    
    
    conclution=pd.DataFrame()
    conclution['Algorithm']=['LDA','QDA','Naive Bayes','Logestic Regression','KNN','Decision Tree','Bagging','Random Forrest','Boosting']
    conclution['MeanAccuracy']=[LDA_Ac_mean,QDA_Ac_mean,NaiveBayes_Ac_mean,LogesticRegression_Ac_mean,KNN_Ac_mean,DecisionTree_Ac_mean,Bagging_Ac_mean,RandomForrest_Ac_mean,Boosting_Ac_mean]
    conclution['STDAccuracy']=[LDA_Ac_std,QDA_Ac_std,NaiveBayes_Ac_std,LogesticRegression_Ac_std,KNN_Ac_std,DecisionTree_Ac_std,Bagging_Ac_std,RandomForrest_Ac_std,Boosting_Ac_std]

    return conclution.sort_values('MeanAccuracy',ascending=False)
