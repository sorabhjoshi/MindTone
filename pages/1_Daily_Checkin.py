"""Daily voice mood check-in: shows a randomly-chosen prompt sentence,
records/accepts audio, detects emotion. This is emotion tracking only —
long-term pattern analysis lives on the History & Trends page, computed
from your accumulated check-in history."""
import datetime

import streamlit as st

from db import get_checkins_for_date, get_session, insert_checkin
from model_loader import analyze_audio_bytes
from prompts import get_random_prompt
from theme import EMOTION_COLORS, animated_bar, hero_banner, inject_theme, stat_card

st.set_page_config(page_title='Daily Mood Check-in', page_icon='🎙️')
inject_theme()

if st.session_state.get('user_id') is None:
    st.warning('Please log in first.')
    st.stop()

hero_banner('Daily Mood Check-in',
            'This tracks your detected speech emotion only — long-term patterns show up on '
            'History & Trends once you have enough check-ins.', '🎙️')

today = datetime.date.today()
if 'prompt_sentence' not in st.session_state:
    st.session_state.prompt_sentence = get_random_prompt()
prompt_sentence = st.session_state.prompt_sentence

session = get_session()
todays_checkins = get_checkins_for_date(session, st.session_state.user_id, today)
session.close()

if todays_checkins:
    st.markdown(f"""
    <div class="glass-card fade-in" style="border-color:rgba(34,211,238,0.3); margin-bottom:1rem;">
        <span style="color:#67E8F9;">✅ You've checked in {len(todays_checkins)} time(s) today.</span>
        <span style="color:#94A3B8;"> Check in again any time — every entry is kept.</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="glass-card fade-in-1" style="margin-bottom:1rem;">
    <p class="badge">Read this sentence out loud</p>
    <h3 style="margin-top:0.6rem;">"{prompt_sentence}"</h3>
</div>
""", unsafe_allow_html=True)

if st.button('🔀 Give me a different sentence'):
    st.session_state.prompt_sentence = get_random_prompt(exclude=prompt_sentence)
    st.rerun()

st.markdown('<div class="fade-in-2">', unsafe_allow_html=True)
st.markdown('#### Record your check-in')
audio_value = st.audio_input('Record yourself reading the sentence above')

with st.expander("Microphone not working? Upload a file instead"):
    uploaded_file = st.file_uploader('Upload a short audio file',
                                      type=['wav', 'mp3', 'm4a', 'ogg', 'flac'])
st.markdown('</div>', unsafe_allow_html=True)

audio_bytes, audio_ext = None, '.wav'
if audio_value is not None:
    audio_bytes = audio_value.getvalue()
    audio_ext = '.wav'
elif uploaded_file is not None:
    audio_bytes = uploaded_file.getvalue()
    audio_ext = '.' + uploaded_file.name.rsplit('.', 1)[-1].lower()

if audio_bytes is not None:
    st.audio(audio_bytes)
    if st.button('✨ Analyze and save', type='primary', width='stretch'):
        with st.spinner('Analyzing your voice ...'):
            emotion_result = analyze_audio_bytes(audio_bytes, suffix=audio_ext)

        session = get_session()
        try:
            insert_checkin(session, st.session_state.user_id, today, prompt_sentence,
                            emotion_result)
        finally:
            session.close()

        st.session_state.prompt_sentence = get_random_prompt(exclude=prompt_sentence)

        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; margin: 1.2rem 0;">
            <span class="badge" style="background:linear-gradient(90deg,#34D39944,#22D3EE44); border-color:#34D39955; color:#A7F3D0;">
                ✅ Saved
            </span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            stat_card('Detected Emotion',
                       f"{emotion_result['predicted_emotion']} "
                       f"({emotion_result['confidence']:.0%})")
        with col2:
            stat_card('Uncertainty', f"{emotion_result['uncertainty']:.2f}")

        st.write('')
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('##### Full probability breakdown')
        for i, (em, p) in enumerate(
                sorted(emotion_result['class_probabilities'].items(), key=lambda x: -x[1])):
            animated_bar(f'{em}', p, EMOTION_COLORS.get(em, '#8B5CF6'), delay_ms=i * 80)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write('')
        st.page_link('pages/2_History_and_Trends.py',
                      label='See your long-term mood patterns →', icon='📈')
