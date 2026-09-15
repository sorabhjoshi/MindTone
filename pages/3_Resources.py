"""Support resources. Shown as a real page (not just a caption) because
this app surfaces mental-health-related flags to people — if it ever
tells someone their indicators look concerning, it should also point
them toward real help, not just a number."""
import streamlit as st

from theme import hero_banner, inject_theme

st.set_page_config(page_title='Resources', page_icon='🤝')
inject_theme()

hero_banner('Support Resources', 'Real help is always more important than a number.', '🤝')

st.markdown("""
<div class="glass-card fade-in-1" style="border-color:rgba(251,191,36,0.4); margin-bottom:1.2rem;">
    <strong style="color:#FCD34D;">⚠️ This app is not a diagnostic or clinical tool.</strong>
    <p style="margin:0.4rem 0 0 0;">It estimates mental-health-related indicators heuristically
    from speech emotion, for self-reflection and tracking only. It cannot replace a
    conversation with a real professional.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card fade-in-2" style="margin-bottom:1.2rem;">
    <h4>🆘 If you're in crisis right now</h4>
    <p>If you're thinking about suicide or self-harm, please reach out immediately:</p>
    <ul>
        <li><strong>US</strong>: Call or text <strong>988</strong> (Suicide &amp; Crisis Lifeline), available 24/7</li>
        <li><strong>India</strong>: Call iCall at <strong>9152987821</strong> (Mon-Sat, 10am-8pm), or AASRA at <strong>+91-9820466726</strong> (24/7)</li>
        <li><strong>UK</strong>: Call Samaritans at <strong>116 123</strong> (24/7)</li>
        <li><strong>Elsewhere</strong>: <a href="https://findahelpline.com" style="color:#67E8F9;">findahelpline.com</a> lists crisis lines by country</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card fade-in-3" style="margin-bottom:1.2rem;">
    <h4>🤝 Ongoing support</h4>
    <ul>
        <li>Talk to a doctor, therapist, or counselor — many universities and workplaces offer free or low-cost counseling</li>
        <li>Talk to someone you trust — a friend, family member, or mentor</li>
        <li>If cost or access is a barrier, community health centers and some nonprofits offer sliding-scale or free mental health services</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card fade-in-4">
    <h4>📋 About what this app measures</h4>
    <p>This app detects speech emotion from a short recording, using a model trained on
    acted/performed emotional speech, which has real, known limits generalizing to natural
    conversational speech. The History &amp; Trends page looks for sustained patterns across
    many check-ins over weeks (not a single reading), which is more meaningful than any one
    day, but it's still a heuristic pattern observation, not a validated clinical measure.
    Treat any flagged pattern as a prompt to reflect — and if it matches how you've actually
    been feeling, talk to someone.</p>
</div>
""", unsafe_allow_html=True)
