"""Account settings: view basic profile info, change password."""
import streamlit as st

from auth import change_password
from db import get_session, get_user_by_username
from theme import hero_banner, inject_theme

st.set_page_config(page_title='Account', page_icon='⚙️')
inject_theme()

if st.session_state.get('user_id') is None:
    st.warning('Please log in first.')
    st.stop()

hero_banner('Account', 'Your profile and security settings.', '⚙️')

session = get_session()
user = get_user_by_username(session, st.session_state.username)
username, email, created_at = user.username, user.email, user.created_at
session.close()

st.markdown(f"""
<div class="glass-card fade-in-1" style="margin-bottom:1.2rem;">
    <p style="margin:0.2rem 0;"><strong style="color:#F1F5F9;">Username:</strong> {username}</p>
    <p style="margin:0.2rem 0;"><strong style="color:#F1F5F9;">Email:</strong> {email}</p>
    <p style="margin:0.2rem 0;"><strong style="color:#F1F5F9;">Member since:</strong> {created_at.strftime('%B %d, %Y')}</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="glass-card fade-in-2">', unsafe_allow_html=True)
st.markdown('#### 🔒 Change password')
with st.form('change_password_form'):
    current_password = st.text_input('Current password', type='password')
    new_password = st.text_input('New password', type='password')
    confirm_password = st.text_input('Confirm new password', type='password')
    submitted = st.form_submit_button('Update password')

if submitted:
    if new_password != confirm_password:
        st.error("New passwords don't match.")
    else:
        success, error = change_password(st.session_state.user_id, current_password, new_password)
        if success:
            st.success('Password updated.')
        else:
            st.error(error)
st.markdown('</div>', unsafe_allow_html=True)

st.write('')
if st.button('Log out', type='secondary'):
    st.session_state.user_id = None
    st.session_state.username = None
    st.rerun()
