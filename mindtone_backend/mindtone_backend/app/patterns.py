"""Derives long-term mood pattern observations from a user's actual
accumulated check-in history — not a single reading run through a
formula, and not a self-report form. Every check-in here is a real
model prediction on real recorded speech; this module just looks at the
pattern across many of them over time.

Be clear-eyed about what this still is: grouping emotions into
'low-mood'/'anxious'/'positive' buckets and flagging a sustained skew is
still a heuristic — there's no way to turn accumulated classifier outputs
into a risk statement without SOME rule doing the mapping. What's
different from the old per-reading score: it requires a real minimum
amount of data before saying anything, it reports plain qualitative
observations instead of a fake-precise percentage-as-diagnosis, and one
noisy day can't dominate a multi-week pattern.
"""
import pandas as pd

MIN_CHECKINS_FOR_PATTERN = 8

# This grouping is itself a judgment call, not a clinical mapping —
# documented here so it's easy to find and question.
LOW_MOOD_EMOTIONS = {'Sad'}
ANXIOUS_EMOTIONS = {'Fear'}
TENSE_EMOTIONS = {'Angry', 'Disgust'}
POSITIVE_EMOTIONS = {'Happy'}

LOW_MOOD_THRESHOLD = 0.40
ANXIOUS_THRESHOLD = 0.35
TENSE_THRESHOLD = 0.40


def summarize_mood_patterns(df: pd.DataFrame, window: int = 30) -> dict:
    """df needs a 'date'/'timestamp' (sorted ascending) and 'emotion'
    column (the predicted dominant emotion per check-in). Returns a dict
    with counts, percentages, qualitative flags, and a trend direction —
    or a 'insufficient_data' flag if there aren't enough check-ins yet."""
    recent = df.tail(window)
    n = len(recent)

    if n < MIN_CHECKINS_FOR_PATTERN:
        return {
            'insufficient_data': True,
            'n_checkins': n,
            'min_required': MIN_CHECKINS_FOR_PATTERN,
        }

    counts = recent['emotion'].value_counts()
    total = len(recent)
    pct = (counts / total).to_dict()

    low_mood_pct = sum(pct.get(e, 0.0) for e in LOW_MOOD_EMOTIONS)
    anxious_pct = sum(pct.get(e, 0.0) for e in ANXIOUS_EMOTIONS)
    tense_pct = sum(pct.get(e, 0.0) for e in TENSE_EMOTIONS)
    positive_pct = sum(pct.get(e, 0.0) for e in POSITIVE_EMOTIONS)

    flags = []
    if low_mood_pct >= LOW_MOOD_THRESHOLD:
        flags.append({
            'label': 'Sustained low-mood pattern',
            'detail': (f'{low_mood_pct:.0%} of your last {total} check-ins showed Sad as '
                       'the dominant emotion. A sustained pattern like this can sometimes '
                       'go along with low mood or depression — worth reflecting on whether '
                       "that matches how you've actually been feeling."),
        })
    if anxious_pct >= ANXIOUS_THRESHOLD:
        flags.append({
            'label': 'Sustained anxious pattern',
            'detail': (f'{anxious_pct:.0%} of your last {total} check-ins showed Fear as '
                       'the dominant emotion. This kind of pattern can sometimes go along '
                       'with anxiety.'),
        })
    if tense_pct >= TENSE_THRESHOLD:
        flags.append({
            'label': 'Sustained tension/irritability pattern',
            'detail': (f'{tense_pct:.0%} of your last {total} check-ins showed Angry or '
                       'Disgust as the dominant emotion.'),
        })

    # Trend: compare the earlier half of the window against the later half
    mid = n // 2
    earlier_negative = recent.iloc[:mid]['emotion'].isin(
        LOW_MOOD_EMOTIONS | ANXIOUS_EMOTIONS | TENSE_EMOTIONS).mean()
    later_negative = recent.iloc[mid:]['emotion'].isin(
        LOW_MOOD_EMOTIONS | ANXIOUS_EMOTIONS | TENSE_EMOTIONS).mean()
    delta = later_negative - earlier_negative
    if abs(delta) < 0.10:
        trend = 'Stable'
    elif delta > 0:
        trend = 'Trending toward more negative-emotion check-ins'
    else:
        trend = 'Trending toward fewer negative-emotion check-ins'

    return {
        'insufficient_data': False,
        'n_checkins': total,
        'emotion_pct': pct,
        'low_mood_pct': low_mood_pct,
        'anxious_pct': anxious_pct,
        'tense_pct': tense_pct,
        'positive_pct': positive_pct,
        'flags': flags,
        'trend': trend,
        'trend_delta': round(float(delta), 3),
    }
