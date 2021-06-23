import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pyttsx3
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle
import seaborn as sns


from sklearn.metrics import f1_score
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
model = pickle.load(open('model_rf.sav', 'rb'))
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")
st.title("Friend Recommendation App")

def main():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df1 = pd.read_csv(uploaded_file)
        
    
        df = pd.read_csv('web_indi.csv')
        test_csv=df1

       
        st.write("let's see  test data")
        st.write(df)
        
        
        subgraph=nx.read_edgelist('web_indi1.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
        
        pos=nx.spring_layout(subgraph)
        st.write(nx.info(subgraph))
        
        image = Image.open('graph.png')
        st.image(image, caption='Graph of Training Data')
        
        st.sidebar.selectbox("Feedback for our website",(1,2,3,4,5))
        
        
        sug=st.sidebar.text_input("Any suggestions regarding our page")
        st.write('Here We dont have enough features to train model , so we need to do feature engineering')
        st.write('After Feature Engineering')
        st.write(test_csv)
        st.write('Extracted Features are\
                 1)INDICATOR LINK (class label)- used to indicate whethere there is a link between two users.')
        st.write('2)JACCARD SIMILARITY - Jaccard distance is nothing but a measure of similarity between two data nodes ranging from 0% to 100%. As the\
                   Jaccard distance increases then there is a high chance of\
                   existing edge between the two nodes, It can be used both for followees and followers.')
        st.write('3)COSINE-DISTANCE(OTSUKA–OCHIAICOEFFICIENT) - Otsuka-Ochiai coefficient is nothing but an intersection of\
                   the no of elements to the square root of the no of elements.')
        st.write('4)INTER FOLLOWERS AND INTER FOLLOWEES - Common followers and followees between two users.')
        st.write('5)FOLLOWS BACK - if a user is following someone and getting followed back by each and ever user than there is higher chance of each and every one following back.')
        st.write('6)PAGE RANK - PageRank is an algorithm that was designed to rank the\
                   importance of web pages. Given a directed graph the PageRank algorithm will give each vertex (Ui ) a score.\
                   The score represents the importance of the vertex in the directed graph.')
        st.write('7)ADAR INDEX - Adar index or Adamic index is nothing but an inverted\
                  sum of degrees of common neighbors for given two vertices.If  user is a normal person adar index will be high , if a user is a popular person than adar index will be low.')
        st.write('8)SINGULAR VALUE DECOMPOSITION – For factorization of matrix into singular values and\
                   singular vectors SVD (Singular Value Decomposition)\
                   algorithm is used. SVD features for both source and destination nodes.')
        st.write('9) WEIGHT FEATURES – An edge weight value was calculated between nodes in\
                    order to find the similarity of nodes. As the neighbor count\
                    goes up edge weight decreases. Intuitively, consider one\
                    million people following a celebrity on a social network\
                    then chances are most of them never met each other or the\
                    celebrity. Whereas on the other hand, if a user has 30\
                    contacts in his/her social network, the chances are higher\
                    that most of them know each other.')
        st.write('10)SHORTEST PATH - if the shortest path is greater than 2 and it should not be much larger .If the path is larger, than there might be less chance of becoming friends and if path is less than there are high chances of becoming friends')
        
                 

        y_test = test_csv.indicator_link
        
        test_csv.drop(['indicator_link'],axis=1,inplace=True)
        y_test_pred = model.predict(test_csv)
        if st.checkbox("Predict", key="A"):
            score=f1_score(y_test,y_test_pred)
            st.write(classification_report(y_test,y_test_pred))
            st.write("Accuracy:",accuracy_score(y_test,y_test_pred))
                
            st.write('Test f1 score:',score)
        st.write("Heat map for Confusion Martix , Precision and Recall")
        
        if st.checkbox("Confusion Martix", key="B"):
            C = confusion_matrix(y_test, y_test_pred,labels=[1,0])
                
            A =(((C.T)/(C.sum(axis=1))).T)
                
            B =(C/C.sum(axis=0))
            plt.figure(figsize=(30,10))
                
            labels = [0,1]
            # representing A in heatmap format
            cmap=sns.light_palette("darkblue")
            plt.subplot(1, 3, 1)
            sns.heatmap(C, annot=True,cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
                
            plt.xlabel('Predicted Class')
            plt.ylabel('Original Class')
            plt.title("Confusion matrix")
                    
            plt.subplot(1, 3, 2)
            sns.heatmap(B, annot=True,cmap=cmap,  fmt=".3f", xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted Class')
            plt.ylabel('Original Class')
            plt.title("Precision matrix")
                    
            plt.subplot(1, 3, 3)
            # representing B in heatmap format
            sns.heatmap(A, annot=True,cmap=cmap, fmt=".3f",xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted Class')
            plt.ylabel('Original Class')
            plt.title("Recall matrix")
                    
            st.pyplot()
        
        st.write('Click on the button to check the ROC-Curve')
        if st.checkbox("ROC-Curve", key="C"):
            from sklearn.metrics import roc_curve, auc
            fpr,tpr,ths = roc_curve(y_test,y_test_pred)
            auc_sc = auc(fpr, tpr)
            plt.figure(figsize=(10,5))
            plt.plot(fpr, tpr, color='navy',label='ROC curve (area = %0.2f)' % auc_sc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic with test data')
            plt.legend()
            st.pyplot()
        st.write('Feature Importance shows the  most important features  which are useful for training the model.')
        if st.checkbox("Feature Importance", key="D"):
            features = test_csv.columns
            importances = model.feature_importances_
            indices = (np.argsort(importances))[-20:]
            plt.figure(figsize=(10,12))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='r', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.ylabel('Important Features')
            st.pyplot()
        st.write("Here our task is to recommend another person,User in the first place might accept this recommendation or not")
        st.write("you can check how many members are actually recommended to the other user")
        
        if st.checkbox('Recommended points'):
            
            for i in range(len(y_test)):
                if y_test_pred[i]==0 and y_test[i]==1:
                   
                   
            
                    st.write(test_csv.iloc[i,0:2])
        
        
       
            
if __name__=='__main__':
    main()
