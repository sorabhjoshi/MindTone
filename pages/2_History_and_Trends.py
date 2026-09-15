"""History & Trends: long-term mood patterns computed from your actual
check-in history (not a single-reading formula, not a form), plus the
underlying emotion data itself."""
import datetime

import altair as alt
import pandas as pd
import streamlit as st

from db import get_session, get_user_checkins
from patterns import summarize_mood_patterns
from theme import EMOTION_COLORS, hero_banner, inject_theme, stat_card

st.set_page_config(page_title='History & Trends', page_icon='📈', layout='wide')
inject_theme()

if st.session_state.get('user_id') is None:
    st.warning('Please log in first.')
    st.stop()

hero_banner('History & Trends', 'Your patterns, from your own data over time.', '📈')

# Dark-mode Altair theme — Altair defaults to a white background, which
# clashes hard with the app's dark theme otherwise.
@alt.theme.register('dark_theme', enable=True)
def dark_altair_theme():
    return {
        'config': {
            'background': 'transparent',
            'view': {'stroke': 'transparent'},
            'axis': {'labelColor': '#94A3B8', 'titleColor': '#CBD5E1',
                     'gridColor': 'rgba(255,255,255,0.06)', 'domainColor': 'rgba(255,255,255,0.15)'},
            'legend': {'labelColor': '#CBD5E1', 'titleColor': '#CBD5E1'},
        }
    }

RANGE_OPTIONS = {'Last 30 days': 30, 'Last 90 days': 90, 'Last 365 days': 365, 'All time': None}
range_choice = st.selectbox('Time range', list(RANGE_OPTIONS.keys()), index=1)
days = RANGE_OPTIONS[range_choice]
since = (datetime.date.today() - datetime.timedelta(days=days)) if days else None

session = get_session()
checkins = get_user_checkins(session, st.session_state.user_id, since=since)
session.close()

if not checkins:
    st.info("No check-ins yet in this range. Head to **Daily Check-in** to record one.")
    st.stop()

df = pd.DataFrame([{
    'date': c.date, 'timestamp': c.created_at, 'emotion': c.predicted_emotion,
    'confidence': c.confidence,
    'Angry': c.prob_angry, 'Disgust': c.prob_disgust, 'Fear': c.prob_fear,
    'Happy': c.prob_happy, 'Neutral': c.prob_neutral, 'Sad': c.prob_sad,
} for c in checkins]).sort_values('timestamp').reset_index(drop=True)

emotion_cols = list(EMOTION_COLORS.keys())

# ── Section 1: long-term pattern detection ────────────────────────────
st.markdown('<div class="fade-in">', unsafe_allow_html=True)
st.markdown('## 🧭 Long-term mood patterns')
st.caption(
    "Computed from your actual check-in history — not a single reading, not a form. "
    "This is still a heuristic — treat it as a prompt to reflect, not a diagnosis.")

pattern = summarize_mood_patterns(df, window=min(len(df), 30))

if pattern['insufficient_data']:
    st.info(f"You've done {pattern['n_checkins']} check-in(s) so far — need at least "
            f"{pattern['min_required']} before a pattern means anything. Keep checking in.")
else:
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card('Check-ins Analyzed', str(pattern['n_checkins']), 'fade-in-1')
    with c2: stat_card('Low-mood (Sad)', f"{pattern['low_mood_pct']:.0%}", 'fade-in-2')
    with c3: stat_card('Anxious (Fear)', f"{pattern['anxious_pct']:.0%}", 'fade-in-3')
    with c4: stat_card('Positive (Happy)', f"{pattern['positive_pct']:.0%}", 'fade-in-4')

    st.write('')
    st.markdown(f'<p class="badge">Trend: {pattern["trend"]}</p>', unsafe_allow_html=True)
    st.write('')

    if pattern['flags']:
        for f in pattern['flags']:
            st.markdown(f"""
            <div class="glass-card fade-in" style="border-color:rgba(251,191,36,0.4); margin-bottom:0.6rem;">
                <strong style="color:#FCD34D;">⚠️ {f['label']}</strong>
                <p style="margin:0.3rem 0 0 0; font-size:0.88rem;">{f['detail']}</p>
            </div>
            """, unsafe_allow_html=True)
        st.caption('If any of this reflects how you\'ve actually been feeling, consider '
                   'talking to a mental health professional or someone you trust — see '
                   'the Resources page.')
    else:
        st.markdown("""
        <div class="glass-card fade-in" style="border-color:rgba(52,211,153,0.4);">
            <span style="color:#6EE7B7;">✅ No sustained concerning pattern detected in this window.</span>
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.write('')
st.markdown('---')

# ── Section 2: streak/summary + underlying emotion data ───────────────
st.markdown('<div class="fade-in-1">', unsafe_allow_html=True)
st.markdown('## 📅 Check-in history')

all_dates = sorted(set(c.date for c in checkins))
today = datetime.date.today()
streak = 0
cursor = today
while cursor in all_dates:
    streak += 1
    cursor -= datetime.timedelta(days=1)

c1, c2, c3, c4 = st.columns(4)
with c1: stat_card('Check-ins in Range', str(len(checkins)))
with c2: stat_card('Current Streak', f'{streak} day{"s" if streak != 1 else ""}')
with c3: stat_card('Most Common', df['emotion'].mode().iloc[0])
with c4: stat_card('Avg. Confidence', f"{df['confidence'].mean():.0%}")
st.markdown('</div>', unsafe_allow_html=True)

st.write('')
st.markdown('<div class="glass-card fade-in-2">', unsafe_allow_html=True)
st.markdown('#### Emotion probabilities over time')
melted = df.melt(id_vars=['timestamp'], value_vars=emotion_cols,
                  var_name='Emotion', value_name='Probability')
chart = alt.Chart(melted).mark_line(point=True).encode(
    x=alt.X('timestamp:T', title='Check-in time'),
    y=alt.Y('Probability:Q', scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('Emotion:N',
                     scale=alt.Scale(domain=emotion_cols,
                                      range=[EMOTION_COLORS[e] for e in emotion_cols])),
    tooltip=['timestamp:T', 'Emotion:N', 'Probability:Q'],
).properties(height=350)
st.altair_chart(chart, width='stretch')
st.markdown('</div>', unsafe_allow_html=True)

st.write('')
st.markdown('<div class="glass-card fade-in-3">', unsafe_allow_html=True)
st.markdown('#### Emotion distribution')
dist_df = df['emotion'].value_counts().reset_index()
dist_df.columns = ['Emotion', 'Count']
bar_chart = alt.Chart(dist_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
    x=alt.X('Emotion:N', sort='-y'),
    y='Count:Q',
    color=alt.Color('Emotion:N',
                     scale=alt.Scale(domain=emotion_cols,
                                      range=[EMOTION_COLORS[e] for e in emotion_cols]),
                     legend=None),
).properties(height=300)
st.altair_chart(bar_chart, width='stretch')
st.markdown('</div>', unsafe_allow_html=True)

st.write('')
with st.expander('All check-ins in range (raw data)'):
    st.dataframe(
        df.sort_values('timestamp', ascending=False).style.format({'confidence': '{:.0%}'}),
        width='stretch', hide_index=True)
    st.download_button('Download as CSV', df.to_csv(index=False),
                        file_name='mood_checkins.csv', mime='text/csv')
