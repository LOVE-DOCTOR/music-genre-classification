import lzma
import pickle
from streamlit import text_input, form, form_submit_button, markdown, write, success, button, set_page_config

set_page_config(page_title='The Music Genre Detector', layout='wide')
write("MUSIC GENRE CLASSIFICATION")

write("Paste your lyrics and I'll tell you if it's rap or not")


with form(key='music'):
    lyrics = text_input("Paste the whole lyrics of the song here")
    submit_button = form_submit_button(label='Submit')

if submit_button:
    success('Just a moment')

if button('Predict now'):
    vector_ = 'vector.xz'
    model_ = 'logreg.xz'

    with lzma.open(model_, 'rb') as f:
        model = pickle.load(f)

    with lzma.open(vector_, 'rb') as g:
        vector = pickle.load(g)

    vectorized = vector.transform([lyrics])
    result = model.predict(vectorized)

    if result[0] == 0:
        success(f"That's some sick rap lyrics there")
    else:
        success(f"What I see here is definitely not rap")
