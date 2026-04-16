## Target agent
- Name: {{ name }} ({{ adopter_tier }})
- Current overall product sentiment: {{ current_sentiment | signed_two_dp }}  (range -1.0 to +1.0)
- Current purchase intent: {{ current_intent | two_dp }}  (range 0.0 to 1.0)
- Recent beliefs about the product:
{{ beliefs_block }}

## Incoming WOM
- From contact with trust = {{ trust | two_dp }}  (range 0.0 to 1.0)
- Declared sentiment: {{ sentiment }}
- Message: "{{ message }}"

## Task
Decide how the target's sentiment and intent should move, and what belief
they now hold.  Ground-rules:
  - Sentiment delta must be small (magnitude ≤ 0.30) unless trust is high AND the message strongly contradicts existing beliefs.
  - Intent delta magnitude ≤ 0.20.
  - If the target has no prior beliefs and the source has low trust, deltas should be small.
  - Negative sentiment from a trusted source should move the needle more than equally-intense positive sentiment.

Respond with JSON:
{
  "sentiment_delta": -1.0 to 1.0,
  "intent_delta": -1.0 to 1.0,
  "new_belief_predicate": "short claim the target now holds, e.g. 'has reliable battery'",
  "belief_confidence": 0.0 to 1.0,
  "memory_importance": 0.0 to 1.0,
  "memory_valence": -1.0 to 1.0,
  "reasoning": "one short sentence"
}
