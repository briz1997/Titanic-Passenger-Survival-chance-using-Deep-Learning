import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival Chance in Titanic")

pclass = st.slider("Enter the passenger class",1,3)

sex = st.selectbox('Enter Passenger gender',['male','female'])

sibsp = st.slider("Enter passenger total number of slibing and spouse",1,8)

parce = st.slider("Enter passenger total number of Parents and childerns",0,6)

fare = st.number_input('Enter the fare of the passenger')

embarked = st.selectbox('Enter the passenger station from where there started',
                        ['Chebourg','Queenstown','Southampton'])

data = pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,
               'Parch':parce,'Fare':fare,'Embarked':embarked}])

model = load_model('ann_model.h5', compile=False)

with open("label.pkl",'rb') as file:
    label = pickle.load(file)

with open('onehot.pkl','rb') as file:
    onehot = pickle.load(file)

with open('scale.pkl','rb') as file:
    scale = pickle.load(file)

data['Sex'] = label.transform(data['Sex'])

Embarked = onehot.transform(data[['Embarked']])

Embarked = pd.DataFrame(Embarked,columns=onehot.get_feature_names_out())

data = pd.concat([data.drop(columns = ['Embarked']),Embarked],axis = 1)

data[['Pclass','SibSp','Parch','Fare']] = scale.transform(data[['Pclass','SibSp','Parch','Fare']])

y = model.predict(data)

y = y[0][0]

def chance(y):
    if y>0.5:
        return "The Passenger will survive in titanic Journey"
    else:
        return "The Passenger wont servied in titanic journey"

if st.button('Predict the serviver'):
    st.write('Probability of serviver passenger are : ',y)
    st.write(chance(y))


