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
import xgboost as xgb
from joblib import load
import seaborn as sns


from sklearn.metrics import f1_score
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
model = joblib.load('random_forest.joblib')
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")
st.title("Facebook Friend Recommendation App")
st.write(""" An app that recommends  users based on edges without feature engineering """)
def main():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df1 = pd.read_csv(uploaded_file)
        st.write(df1[:10])
    
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
        y_test = test_csv.indicator_link
        
        test_csv.drop(['indicator_link'],axis=1,inplace=True)
        def pred():
            if st.button("Predict"):
                
                y_test_pred = model.predict(test_csv)
                
                score=f1_score(y_test,y_test_pred)
                st.write(classification_report(y_test,y_test_pred))
                st.write("Accuracy:",accuracy_score(y_test,y_test_pred))
                
                st.write('Test f1 score:',score)
        pred()
        st.write("Heat map for Confusion Martix , Precision and Recall")
        def but():
            if st.button("Confusion Matrix"):
                y_test_pred = model.predict(test_csv)        
                C = confusion_matrix(y_test, y_test_pred)
                
                A =(((C.T)/(C.sum(axis=1))).T)
                
                B =(C/C.sum(axis=0))
                plt.figure(figsize=(30,10))
                
                labels = [0,1]
                # representing A in heatmap format
                
                
                
                cmap=sns.light_palette("blue")
                plt.subplot(1, 3, 1)
                sns.heatmap(C, annot=True,cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
                
                plt.xlabel('Predicted Class')
                plt.ylabel('Original Class')
                plt.title("Confusion matrix")
                    
                plt.subplot(1, 3, 2)
                sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
                plt.xlabel('Predicted Class')
                plt.ylabel('Original Class')
                plt.title("Precision matrix")
                    
                plt.subplot(1, 3, 3)
                    # representing B in heatmap format
                sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
                plt.xlabel('Predicted Class')
                plt.ylabel('Original Class')
                plt.title("Recall matrix")
                    
                st.pyplot()
        but()
        st.write("Here our task is to recommend,User in the first place might accept this recommendation or not")
        st.write("you can check how many members are actually recommended to the other user , click the below button")
        def recomm():
            if st.button('Recommended points'):
                
                y_test_pred = model.predict(test_csv)        
                C = confusion_matrix(y_test, y_test_pred)
                st.write(y_test_pred)
                l=[]
                for i in range(len(y_test)):
                    if y_test_pred[i]==1 and y_test[i]==0:
                        l.append('FP')
                st.write(len(l))
        recomm()
        st.balloons()
        audio=pyttsx3.init()
        audio.setProperty('rate', 200) 
        audio.setProperty('volume', 0.7) 
        audio.say("Thank you for using this page")
        audio.runAndWait()
            
if __name__=='__main__':
    main()
