import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

st.write("""
#  Mortaage Price Prediction  detection app
This app predicts the price **Mean_mortage_price**!
""")


st.write('---')


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')


new_columns=['lat','lng','type','state','pop','debt','rent_mean','hi_mean','family_mean','hc_mean','home_equity','second_mortgage','pct_own']
def user_input_features():
    typee = st.sidebar.selectbox('type',(1,2,3,4,5,6))  
    state=st.sidebar.slider('state',0,51)
    pop=st.sidebar.slider('pop',0,53812)
    Rent_mean=st.sidebar.slider('rent_mean',117.150000,3962.342290)
    hi_mean=st.sidebar.slider('hi_mean',0.000000,2.420000)
    family_mean=st.sidebar.slider('family_mean',5374.842520,242857.142900)
    hc_mean=st.sidebar.slider('hc_mean',53.594610,1700.179110)
    home_equity=st.sidebar.slider('home_equity',0.000000,1.000000)
    second_mortgage=st.sidebar.slider('second_mortgage',0.000000,1.000000)
    debt=st.sidebar.slider('debt',0.000000,1.000000)
    pct_own=st.sidebar.slider('pct_own',0.000000,1.000000)
  
       
    data = {
        'typee': typee,
        'state':state,
            'pop':pop,
            'debt':debt,
            'Rent_mean': Rent_mean,
            'hi_mean':hi_mean,
            'family_mean': family_mean,
            'hc_mean': hc_mean,
            'home_equity':home_equity,
            'second_mortgage':second_mortgage,
            
            'pct_own':pct_own
        
            }   

    features = pd.DataFrame(data, index=[0])

    return features

df = user_input_features()



scaler = StandardScaler()
scaledd_values=scaler.fit_transform(df)
scaledd_df=pd.DataFrame(scaledd_values,columns=df.columns)





load_clf=pickle.load(open('Lineak_RealEstate.pkl', 'rb'))




prediction=load_clf.predict(df)

            



st.header('Specified Input parameters')
st.write(df)
st.write('it has to do with')






st.subheader('Prediction')

st.write(prediction)


# st.subheader('Prediction Proability')

# st.write(prediction_praob)




# explainer = shap.TreeExplainer(load_clf)
# shap_values = explainer.shap_values(df)

# st.header('Feature Importance')
# st.set_option('deprecation.showPyplotGlobalUse', False)
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, df)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# st.set_option('deprecation.showPyplotGlobalUse', False)


# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, df, plot_type="bar")
# st.pyplot(bbox_inches='tight')
