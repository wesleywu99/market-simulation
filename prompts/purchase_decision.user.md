Profile: {{ name }}, {{ age }}yo {{ occupation }}, {{ income_level }} income ({{ income_amount | money }}/mo), {{ adopter_tier }}, [{{ lifestyle_tags }}]
Cognitive style: {{ cognitive_style }}
Goals: {{ goals }}
Beliefs: {{ beliefs }}
Memories: {{ memories }}
Social: {{ adopted_friends }} friends bought, {{ network_adoption_rate | pct }} network adoption rate
Product: {{ product_name }} by {{ brand }} | {{ price | money }} | {{ features }} | quality {{ quality | pct }} | brand rep {{ brand_reputation | pct }} | channels: {{ channels }}
Risk: tolerance {{ risk_tolerance | one_dp }} | loss aversion {{ loss_aversion | one_dp }}x | available budget {{ budget | money }}
{% if market_context %}
Market context:
{{ market_context }}
{% endif %}
Timeline: step {{ current_step }} ({{ days_since_launch }} days since product launch, each step = {{ step_duration_days }} days).

Decide BUY / DEFER / REJECT. Output JSON only.
IMPORTANT for `deferred_until`:
  - If decision = "defer", it MUST be an integer equal to {{ min_defer }} through {{ max_defer }} (i.e. current_step + 1 to current_step + 5). It represents a FUTURE SIMULATION STEP, not a calendar year.
  - If decision = "buy" or "reject", set deferred_until to null.

{"decision":"buy|defer|reject","confidence":0.0-1.0,"reasoning":"brief","perceived_attributes":{"relative_advantage":0.0-1.0,"compatibility":0.0-1.0,"complexity":0.0-1.0,"trialability":0.0-1.0,"observability":0.0-1.0},"price_acceptable":true|false,"key_concerns":["..."],"social_influence_weight":0.0-1.0,"deferred_until":null}
