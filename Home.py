"""
MindTone — home page (login / signup). Run with: streamlit run Home.py
"""
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from auth import login, signup
from db import init_db
from theme import feature_card, hero_banner, inject_theme

st.set_page_config(page_title='MindTone', page_icon='🎙️', layout='centered')
inject_theme()
init_db()

if 'user_id' not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = None


def show_landing_and_auth():
    hero_banner(
        'MindTone',
        "A voice check-in that tracks how you're really doing, over weeks — not just today.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        feature_card('🗣️', 'Speak', 'Read a short sentence out loud — takes under a minute.', 'fade-in-1')
    with col2:
        feature_card('📊', 'Track', 'Every check-in is logged. Nothing gets overwritten.', 'fade-in-2')
    with col3:
        feature_card('📈', 'See patterns', 'Real long-term patterns from your own data, over weeks.', 'fade-in-3')

    st.write('')
    st.markdown('<div class="fade-in-4">', unsafe_allow_html=True)
    with st.container(border=False):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        tab_login, tab_signup = st.tabs(['Log in', 'Sign up'])

        with tab_login:
            with st.form('login_form'):
                username = st.text_input('Username', key='login_username')
                password = st.text_input('Password', type='password', key='login_password')
                submitted = st.form_submit_button('Log in', width='stretch')
            if submitted:
                user, error = login(username, password)
                if error:
                    st.error(error)
                else:
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.rerun()

        with tab_signup:
            with st.form('signup_form'):
                new_username = st.text_input('Choose a username', key='signup_username')
                new_email = st.text_input('Email', key='signup_email')
                new_password = st.text_input('Choose a password', type='password', key='signup_password',
                                              help='At least 8 characters, with a letter and a number.')
                confirm_password = st.text_input('Confirm password', type='password', key='signup_confirm')
                submitted = st.form_submit_button('Create account', width='stretch')
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords don't match.")
                else:
                    user, error = signup(new_username, new_email, new_password)
                    if error:
                        st.error(error)
                    else:
                        st.success('Account created — you can log in now.')
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center; font-size:0.78rem; color:#475569; margin-top:1.5rem;'>"
        "Mood is tracked from your speech over time — patterns are derived from your own "
        "accumulated data, not a single reading or a form. This isn't a diagnosis."
        "</p>", unsafe_allow_html=True)


def show_dashboard_home():
    hero_banner(f'Welcome back, {st.session_state.username}', 'What would you like to do today?', '👋')

    col1, col2, col3 = st.columns(3)
    with col1:
        feature_card('🎙️', 'Daily Mood', 'Read a sentence, track your detected emotion.', 'fade-in-1')
        st.page_link('pages/1_Daily_Checkin.py', label='Check in →')
    with col2:
        feature_card('📈', 'History & Trends', 'See your long-term patterns.', 'fade-in-2')
        st.page_link('pages/2_History_and_Trends.py', label='View History →')
    with col3:
        feature_card('🤝', 'Resources', 'Support resources, always available.', 'fade-in-3')
        st.page_link('pages/3_Resources.py', label='View Resources →')

    st.write('')
    st.markdown("""
    <div class="glass-card fade-in-4" style="border-color: rgba(34,211,238,0.3);">
        <strong style="color:#67E8F9;">Note:</strong>
        <span style="color:#CBD5E1;">long-term mood patterns here are derived heuristically from your
        own speech data, not a clinical diagnosis. If you're struggling, please talk to a
        mental health professional or a trusted person.</span>
    </div>
    """, unsafe_allow_html=True)

    st.write('')
    if st.button('Log out'):
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()


if st.session_state.user_id is None:
    show_landing_and_auth()
else:
    show_dashboard_home()
