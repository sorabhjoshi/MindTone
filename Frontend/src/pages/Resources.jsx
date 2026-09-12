import Layout from '../components/Layout';

export default function Resources() {
  return (
    <Layout>
      <h1 className="text-2xl font-bold text-teal-900">🤝 Support Resources</h1>

      <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 p-5">
        <p className="font-semibold text-amber-900">
          This app is not a diagnostic or clinical tool.
        </p>
        <p className="mt-1 text-sm text-amber-900/80">
          It estimates mental-health-related patterns heuristically from speech emotion, for
          self-reflection and tracking only. It cannot replace a conversation with a real
          professional.
        </p>
      </div>

      <section className="mt-6 rounded-2xl border border-teal-100 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-teal-900">If you're in crisis right now</h2>
        <p className="mt-1 text-sm text-teal-900/70">
          If you&rsquo;re thinking about suicide or self-harm, please reach out immediately:
        </p>
        <ul className="mt-3 space-y-2 text-sm text-teal-900/80">
          <li><strong>US:</strong> Call or text <strong>988</strong> (Suicide &amp; Crisis Lifeline), available 24/7</li>
          <li><strong>India:</strong> Call iCall at <strong>9152987821</strong> (Mon-Sat, 10am-8pm), or AASRA at <strong>+91-9820466726</strong> (24/7)</li>
          <li><strong>UK:</strong> Call Samaritans at <strong>116 123</strong> (24/7)</li>
          <li><strong>Elsewhere:</strong> <a href="https://findahelpline.com" target="_blank" rel="noreferrer" className="text-teal-600 underline">findahelpline.com</a> lists crisis lines by country</li>
        </ul>
      </section>

      <section className="mt-4 rounded-2xl border border-teal-100 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-teal-900">Ongoing support</h2>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-teal-900/80">
          <li>Talk to a doctor, therapist, or counselor — many universities and workplaces offer free or low-cost counseling</li>
          <li>Talk to someone you trust — a friend, family member, or mentor</li>
          <li>If cost or access is a barrier, community health centers and some nonprofits offer sliding-scale or free mental health services</li>
        </ul>
      </section>

      <section className="mt-4 rounded-2xl border border-teal-100 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-teal-900">About what this app measures</h2>
        <p className="mt-2 text-sm text-teal-900/80">
          This app detects speech emotion from a short recording, using a model trained on
          acted/performed emotional speech, which has real, known limits generalizing to natural
          conversational speech. The History &amp; Trends page looks for sustained patterns across
          many check-ins over weeks (not a single reading), which is more meaningful than any one
          day, but it's still a heuristic pattern observation, not a validated clinical measure.
          Treat any flagged pattern as a prompt to reflect — and if it matches how you've
          actually been feeling, talk to someone.
        </p>
      </section>
    </Layout>
  );
}
