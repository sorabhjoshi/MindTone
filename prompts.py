"""Prompt sentences read aloud for each mood check-in. A random sentence
each time — not tied to the calendar date — so back-to-back check-ins
(now that multiple per day are allowed) get varied content instead of
reading identical text repeatedly, which would trivially produce similar
emotion readings regardless of the model. All are short, neutral,
phonetically varied, original sentences."""
import random

PROMPT_SENTENCES = [
    "The weather changed quickly after the sun went down.",
    "She placed the book back on the shelf before leaving.",
    "Traffic was lighter than usual on the way to work.",
    "He poured a glass of water and sat by the window.",
    "The train arrived a few minutes ahead of schedule.",
    "They walked along the path until it reached the river.",
    "A soft breeze moved through the open door.",
    "The market was busy with people buying fresh vegetables.",
    "She wrote the date at the top of the page.",
    "The children played outside until it started to rain.",
    "He checked his phone before starting the car.",
    "The coffee was still warm when she picked it up.",
    "A dog barked somewhere down the quiet street.",
    "The meeting was moved to a later time in the afternoon.",
    "She folded the laundry while listening to the radio.",
    "The lights in the hallway flickered for a moment.",
    "He read the letter twice before putting it away.",
    "The bakery on the corner opens early every morning.",
    "They packed their bags the night before the trip.",
    "The clock on the wall was a few minutes slow.",
    "She watered the plants on the balcony before breakfast.",
    "The bridge was closed for repairs over the weekend.",
    "He tied his shoes and stepped out into the cold.",
    "The library stays open later on weekdays.",
    "A plane flew low over the houses near the airport.",
    "She organized her desk before starting the new project.",
    "The bus stopped at every corner along the main road.",
    "He turned off the lights and locked the front door.",
    "The garden looked different after the first snow.",
    "They sat quietly and watched the sunset from the porch.",
]


def get_random_prompt(exclude: str = None) -> str:
    """A fresh, randomly-chosen sentence each call — not tied to the
    calendar date. Avoids immediately repeating whatever sentence was
    just shown (if provided), so back-to-back check-ins get genuinely
    different content rather than reading the same line twice."""
    choices = PROMPT_SENTENCES
    if exclude is not None and len(PROMPT_SENTENCES) > 1:
        choices = [s for s in PROMPT_SENTENCES if s != exclude]
    return random.choice(choices)
