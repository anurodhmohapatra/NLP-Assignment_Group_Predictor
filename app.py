import pickle

import streamlit as st

st.beta_set_page_config(page_title='AM', page_icon=None, layout='centered', initial_sidebar_state='auto')

# To hide hamburger (top right corner) and “Made with Streamlit” footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# load the model from disk
clf = pickle.load(open('clf.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))


def predict(description):
    prediction = clf.predict(cv.transform(description))
    return le.inverse_transform(prediction)[0]


def main():
    st.title("Assignment group prediction for SNOW tickets")
    st.markdown("#### Enter ticket description below")
    description = st.text_area("Description")
    if st.button("Predict"):
        text = [description]
        assignment_group = predict(text)
        st.write("### Assignment group: ")
        st.success(assignment_group)


if __name__ == '__main__':
    main()
