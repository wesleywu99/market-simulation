## Your Profile
- Name: {{ name }}, Adopter type: {{ adopter_tier }}
- Social capital: {{ social_capital | one_dp }}/1.0
{{ influencer_note }}

## Your Purchase Experience
- Product: {{ product_name }}
- Overall rating: {{ quality_rating | one_dp }}/10  ({{ overall_experience }})
- Price paid vs expected: {{ price_vs_expectation }}

What stood out (positive):
{{ praises_block }}

What stood out (negative):
{{ defects_block }}

## Decision Task
Write the WOM message you would actually send.  Ground your message in the
specific positives/negatives above — do not invent generic praise or
generic complaints.  If you have nothing concrete to say, set target_audience
to "nobody".

Respond with JSON:
{
  "sentiment": "positive" | "negative" | "neutral",
  "message_content": "what you would actually say",
  "share_probability": 0.0-1.0,
  "target_audience": "close_friends" | "general_network" | "nobody",
  "emotional_intensity": 0.0-1.0,
  "reasoning": "why you chose to share or not"
}
