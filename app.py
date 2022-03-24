import sklearntrain
import app2
import app1
import pycaret_training
#import app3 # create app3 file
import streamlit as st
def run():
    PAGES = {
    "Sklearn Training": sklearntrain,
    "Sklearn Testing": app2,
    "Pyacret Training": pycaret_training,
    "Pycaret Testing": app1
    #"PycaretNew Testing":app3
     }
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app()

if __name__ == '__main__':
    run()