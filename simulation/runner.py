"""
SimulationRunner — multi-step simulation orchestrator.

Owns the top-level loop:  **evaluate -> WOM generate -> WOM propagate -> metrics**.

All swappable strategies (network topology, WOM propagation) are injected via
``SimulationConfig``, keeping the runner itself strategy-agnostic.
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from agents.belief import BeliefUpdater, LinearBeliefUpdater
from agents.consumer import ConsumerAgent
from agents.experience import BlindBoxExperience, ExperienceSampler
from agents.filters import DecisionFilter, System1Filter
from agents.influencer import DegreeBasedSeeding, InfluencerSeeding
from core.events import Event, EventType
from environment.network import NetworkBuilder, SmallWorldNetwork
from environment.product import Product
from environment.wom import WOMEngine, TrustWeightedWOM, WOMReception
from llm.schemas import PurchaseDecision, PurchaseDecisionOutput, WOMOutput
from simulation.controller import make_agent, _assign_tiers, TIER_TEMPERATURE
from simulation.events import EventSchedule, apply_event
from simulation.population import PopulationSpec
from simulation.metrics import MetricsCollector, StepMetrics
from simulation.trace import NoOpTrace, TraceWriter


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """All knobs for a simulation run.

    Required: *product*.  Everything else has sensible defaults.
    Swap *network_builder* or *wom_engine* to change topology or
    propagation strategy without touching the runner.
    """
    product: Product
    n_agents: int = 25
    n_steps: int = 5
    seed: int = 42

    # Inject strategies -- None -> use built-in defaults
    network_builder: Optional[NetworkBuilder] = None
    wom_engine: Optional[WOMEngine] = None
    decision_filter: Optional[DecisionFilter] = None
    experience_sampler: Optional[ExperienceSampler] = None
    influencer_seeding: Optional[InfluencerSeeding] = None
    belief_updater: Optional[BeliefUpdater] = None

    # Time semantics — maps abstract steps to real calendar time.
    # "1 step = step_duration_days days."  Used for prompt context,
    # budget refresh, and memory-decay calibration.
    step_duration_days: int = 7              # default: 1 step = 1 week
    budget_refresh_interval: int = 4         # refresh every N steps (4 weeks ≈ monthly)

    # Belief decay — applied once per step to prevent sentiment/intent
    # saturation.  Sentiment decays faster (emotion fades) than intent
    # (purchase motivation is stickier).  Set to 1.0 to disable.
    sentiment_decay: float = 0.95
    intent_decay: float = 0.97

    # External event schedule — market shocks, media, advertising.
    # None = closed system (WOM-only dynamics).
    event_schedule: Optional[EventSchedule] = None

    # Market context — free-text category briefing injected into the
    # purchase decision prompt.  Grounds agent decisions in real-world
    # knowledge: competing brands, typical price range, purchase triggers.
    # None = agents decide with no external reference frame.
    market_context: Optional[str] = None

    # Population demographics — overrides the hardcoded defaults in
    # controller.py.  None = use built-in Chinese tech demographics.
    population_spec: Optional[PopulationSpec] = None

    # Tracing — pass a path to enable JSONL decision/WOM logs + step
    # snapshots.  None disables tracing (NoOpTrace is used internally).
    trace_dir: Optional[str] = None
    run_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# SIMULATION REPORT (returned by runner.run())
# ─────────────────────────────────────────────────────────────

@dataclass
class SimulationReport:
    """Immutable summary returned after a completed simulation."""
    product_name: str
    n_agents: int
    n_steps: int
    step_duration_days: int
    network_type: str
    wom_type: str
    filter_type: str
    seeding_type: str
    belief_updater_type: str
    influencer_ids: List[str]
    final_adoption_rate: float
    total_sales: int
    total_revenue: float
    adoption_curve: List[float]
    step_metrics: List[StepMetrics]
    all_events: List[Event]
    tier_adoption: Dict[str, Dict]   # from the final step
    system1_rejects: int = 0
    system1_defers: int = 0
    system2_calls: int = 0
    trace_dir: Optional[str] = None  # populated when tracing is enabled


# ─────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────

class SimulationRunner:
    """Orchestrates a multi-step market simulation.

    Lifecycle
    ---------
    1. ``__init__``: create agents, build social network
    2. ``run()``: execute steps 1…N, return ``SimulationReport``

    Each step has four phases:

    ┌─────────────────────────────────────────────┐
    │  Phase 1  Identify evaluators               │
    │  Phase 2  Purchase decisions (LLM calls)     │
    │  Phase 3  WOM generation + propagation       │
    │  Phase 4  Metrics snapshot                   │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.product = config.product
        self.rng = random.Random(config.seed)

        # Resolve strategies (defaults if not injected)
        self.network_builder = config.network_builder or SmallWorldNetwork(seed=config.seed)
        self.wom_engine = config.wom_engine or TrustWeightedWOM(seed=config.seed)
        self.decision_filter = config.decision_filter or System1Filter()
        self.experience_sampler = config.experience_sampler or BlindBoxExperience()
        self.influencer_seeding = config.influencer_seeding or DegreeBasedSeeding()
        self.belief_updater = config.belief_updater or LinearBeliefUpdater()

        # Trace sink — TraceWriter when enabled, NoOpTrace otherwise.
        if config.trace_dir:
            self.trace = TraceWriter(base_dir=config.trace_dir, run_id=config.run_id)
        else:
            self.trace = NoOpTrace()

        # System 1 telemetry — counted per run
        self.system1_rejects = 0
        self.system1_defers = 0
        self.system2_calls = 0

        # Create agents
        self.agents: List[ConsumerAgent] = self._create_agents()
        self.agents_map: Dict[str, ConsumerAgent] = {a.agent_id: a for a in self.agents}

        # Simulation state
        self.adopted_ids: Set[str] = set()
        self.deferred: Dict[int, Set[str]] = defaultdict(set)  # step -> agent_ids
        self.wom_recipients: Set[str] = set()  # agents that received WOM since last eval
        self.all_events: List[Event] = []
        self.metrics = MetricsCollector()

        # Build social network, then amplify selected agents as KOLs
        self.network_builder.build(self.agents)
        self.influencer_ids: List[str] = self.influencer_seeding.seed(self.agents)

    # ── public interface ──────────────────────────────────────

    def run(self) -> SimulationReport:
        """Execute the full simulation (sync wrapper around ``run_async``)."""
        return asyncio.run(self.run_async())

    async def run_async(self) -> SimulationReport:
        """Execute the full simulation and return a report."""
        self._print_header()

        try:
            for step in range(1, self.config.n_steps + 1):
                self.product.reset_step_counters()
                await self._run_step(step)
                self.trace.write_snapshot(step, self.agents)
        finally:
            self.trace.close()

        return self._build_report()

    # ── step execution ────────────────────────────────────────

    async def _run_step(self, step: int) -> None:
        # Phase 0a: decay sentiment & intent for non-buyers (models "cooling off")
        if step > 1:
            self._apply_belief_decay()

        # Phase 0b: budget refresh on schedule (models monthly income)
        self._maybe_refresh_budgets(step)

        # Phase 0c: apply external events (price changes, media, ads)
        self._apply_scheduled_events(step)

        # Phase 1: identify who evaluates
        evaluators, reason = self._get_evaluators(step)
        self._print_step_header(step, len(evaluators), reason)

        if not evaluators:
            self._record_empty_step(step)
            print("  (no agents to evaluate this step)\n")
            return

        # Phase 2a: System 1 pre-filter (fast, no LLM) ─────────────
        system1_decisions: List[Tuple[ConsumerAgent, PurchaseDecisionOutput, str]] = []
        llm_agents: List[ConsumerAgent] = []

        for agent in evaluators:
            fr = self.decision_filter.check(
                agent=agent,
                product=self.product,
                adopted_ids=self.adopted_ids,
                current_step=step,
            )
            if fr.decision is not None:
                system1_decisions.append((agent, fr.decision, fr.reason))
            else:
                llm_agents.append(agent)

        print(
            f"  System 1: {len(system1_decisions)} short-circuited, "
            f"{len(llm_agents)} to LLM (saved "
            f"{len(system1_decisions)}/{len(evaluators)} calls)",
            flush=True,
        )

        # Phase 2b: System 2 (LLM) decisions — all in parallel ─────
        if llm_agents:
            print(f"  Dispatching {len(llm_agents)} LLM decisions in parallel...", flush=True)
            tasks = [self._decide_with_retry(agent, step) for agent in llm_agents]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            outcomes = []

        decisions: List[Tuple[ConsumerAgent, PurchaseDecisionOutput]] = []
        step_buyers = 0
        step_deferrers = 0
        step_rejecters = 0

        # Apply System 1 results first (so adopted_ids carries into display)
        counter = 0
        total = len(evaluators)

        for agent, decision_output, filter_reason in system1_decisions:
            counter += 1
            p = agent.state.profile
            prefix = (
                f"  [{counter:>2}/{total}] {p.name:<14} "
                f"({p.adopter_tier:<14}) ¥{p.income_amount:>7,.0f}  ..."
            )
            decisions.append((agent, decision_output))
            self.all_events.append(self._system1_event(agent, decision_output, filter_reason, step))

            target_step: Optional[int] = None
            if decision_output.decision == PurchaseDecision.DEFER:
                target_step = self._clamp_deferred(decision_output.deferred_until, step)
                self.deferred[target_step].add(agent.agent_id)
                step_deferrers += 1
                self.system1_defers += 1
                print(f"{prefix} [DEFER*]  (S1:{filter_reason} -> step {target_step})")
            else:  # REJECT
                step_rejecters += 1
                self.system1_rejects += 1
                print(f"{prefix} [REJECT*] (S1:{filter_reason})")

            self.trace.write_decision(
                step=step,
                agent_id=agent.agent_id,
                tier=p.adopter_tier,
                decision=decision_output.decision.value,
                confidence=decision_output.confidence,
                price_acceptable=decision_output.price_acceptable,
                source="system1",
                filter_reason=filter_reason,
                deferred_until=target_step,
                reasoning=decision_output.reasoning,
            )

        # Apply LLM results
        for agent, outcome in zip(llm_agents, outcomes):
            counter += 1
            self.system2_calls += 1
            p = agent.state.profile
            prefix = (
                f"  [{counter:>2}/{total}] {p.name:<14} "
                f"({p.adopter_tier:<14}) ¥{p.income_amount:>7,.0f}  ..."
            )

            if isinstance(outcome, Exception):
                print(f"{prefix} ERROR: {outcome}")
                self.all_events.append(Event(
                    event_type=EventType.STATE_CHANGE,
                    timestamp=step,
                    source=agent.agent_id,
                    content={"error": str(outcome)},
                ))
                continue

            decision_output, event = outcome
            decisions.append((agent, decision_output))
            self.all_events.append(event)

            llm_target_step: Optional[int] = None
            if decision_output.decision == PurchaseDecision.BUY:
                self.adopted_ids.add(agent.agent_id)
                self.product.record_sale(self.product.price)
                step_buyers += 1
                print(f"{prefix} [BUY]     (confidence {decision_output.confidence:.0%})")
            elif decision_output.decision == PurchaseDecision.DEFER:
                llm_target_step = self._clamp_deferred(decision_output.deferred_until, step)
                self.deferred[llm_target_step].add(agent.agent_id)
                step_deferrers += 1
                print(f"{prefix} [DEFER]   (until step {llm_target_step})")
            else:
                step_rejecters += 1
                print(f"{prefix} [REJECT]  (confidence {decision_output.confidence:.0%})")

            self.trace.write_decision(
                step=step,
                agent_id=agent.agent_id,
                tier=p.adopter_tier,
                decision=decision_output.decision.value,
                confidence=decision_output.confidence,
                price_acceptable=decision_output.price_acceptable,
                source="system2",
                filter_reason=None,
                deferred_until=llm_target_step,
                reasoning=decision_output.reasoning,
                key_concerns=decision_output.key_concerns,
                social_influence_weight=decision_output.social_influence_weight,
            )

        # Phase 3: WOM generation and propagation
        wom_generated, wom_received = await self._run_wom_phase(step, decisions)

        # Phase 4: metrics
        sm = self.metrics.record_step(
            step=step,
            total_agents=len(self.agents),
            evaluators=len(evaluators),
            buyers=step_buyers,
            deferrers=step_deferrers,
            rejecters=step_rejecters,
            cumulative_buyers=len(self.adopted_ids),
            wom_generated=wom_generated,
            wom_received=wom_received,
            agents=self.agents,
            adopted_ids=self.adopted_ids,
        )

        self._print_step_footer(sm, wom_generated, wom_received)

    # ── evaluator selection ───────────────────────────────────

    def _get_evaluators(self, step: int) -> Tuple[List[ConsumerAgent], str]:
        """Determine which agents evaluate this step and why."""
        if step == 1:
            # Product launch: everyone is exposed
            agents = [a for a in self.agents]
            return agents, "product launch — all agents"

        evaluator_ids: Set[str] = set()
        reasons = []

        # Deferred agents scheduled for this step
        deferred_now = self.deferred.pop(step, set())
        deferred_eligible = deferred_now - self.adopted_ids
        if deferred_eligible:
            evaluator_ids |= deferred_eligible
            reasons.append(f"{len(deferred_eligible)} deferred")

        # Agents who received WOM since their last evaluation
        wom_eligible = self.wom_recipients - self.adopted_ids
        if wom_eligible:
            evaluator_ids |= wom_eligible
            reasons.append(f"{len(wom_eligible)} WOM-influenced")
        self.wom_recipients.clear()

        agents = [self.agents_map[aid] for aid in evaluator_ids if aid in self.agents_map]
        reason = " + ".join(reasons) if reasons else "none"
        return agents, reason

    # ── WOM phase ─────────────────────────────────────────────

    async def _run_wom_phase(
        self,
        step: int,
        decisions: List[Tuple[ConsumerAgent, PurchaseDecisionOutput]],
    ) -> Tuple[int, int]:
        """Have new buyers generate WOM; propagate through network.

        Returns (wom_messages_generated, wom_receptions).
        """
        new_buyers = [
            (agent, dec) for agent, dec in decisions
            if dec.decision == PurchaseDecision.BUY
        ]
        if not new_buyers:
            return 0, 0

        # Sample each buyer's blind-box experience (deterministic given seed)
        experiences = [
            self.experience_sampler.sample(self.product, agent, self.rng)
            for agent, _ in new_buyers
        ]

        # Generate WOM in parallel
        wom_tasks = [
            agent.generate_wom(
                product=self.product,
                experience=experience,
                price_paid=self.product.price,
                current_step=step,
            )
            for (agent, _), experience in zip(new_buyers, experiences)
        ]
        wom_results = await asyncio.gather(*wom_tasks, return_exceptions=True)

        total_generated = 0
        total_received = 0

        for (agent, _), outcome in zip(new_buyers, wom_results):
            if isinstance(outcome, Exception):
                print(f"    WOM error for {agent.state.profile.name}: {outcome}")
                continue

            wom_output, wom_event = outcome
            if wom_event is None:
                continue

            total_generated += 1
            self.all_events.append(wom_event)

            # Propagate through network (sync — no LLM, just code)
            receptions: List[WOMReception] = self.wom_engine.compute_receptions(
                source_id=agent.agent_id,
                source_graph=agent.state.relationships,
                wom=wom_output,
            )

            update_tasks = []
            reached = 0
            for reception in receptions:
                target = self.agents_map.get(reception.target_id)
                if target is None or target.state.has_purchased:
                    continue

                # Layer 2: queue a belief update (may be LLM-backed).
                update_tasks.append(self.belief_updater.update(
                    target=target,
                    source_id=agent.agent_id,
                    sentiment=wom_output.sentiment.value,
                    message=wom_output.message_content,
                    trust=reception.trust,
                    step=step,
                ))

                self.wom_recipients.add(reception.target_id)
                reached += 1

                self.all_events.append(Event(
                    event_type=EventType.WORD_OF_MOUTH,
                    timestamp=step,
                    source=agent.agent_id,
                    target=reception.target_id,
                    content={
                        "product_id": self.product.product_id,
                        "sentiment": wom_output.sentiment.value,
                        "trust": reception.trust,
                        "reception_strength": reception.reception_strength,
                    },
                ))

                self.trace.write_wom(
                    step=step,
                    source_id=agent.agent_id,
                    target_id=reception.target_id,
                    sentiment=wom_output.sentiment.value,
                    trust=reception.trust,
                    reception_strength=reception.reception_strength,
                    emotional_intensity=wom_output.emotional_intensity,
                    target_audience=wom_output.target_audience.value,
                    message=wom_output.message_content,
                )

            # Run this buyer's reception updates in parallel (LLM-safe).
            if update_tasks:
                update_outcomes = await asyncio.gather(*update_tasks, return_exceptions=True)
                for outcome in update_outcomes:
                    if isinstance(outcome, Exception):
                        print(f"    [warn] belief update failed: {outcome}")

            total_received += reached
            if reached > 0:
                print(
                    f"    WOM: {agent.state.profile.name} shared "
                    f"{wom_output.sentiment.value} experience -> reached {reached} contacts"
                )

        return total_generated, total_received

    # ── belief decay ────────────────────────────────────────

    def _apply_scheduled_events(self, step: int) -> None:
        """Process external events scheduled for this step."""
        schedule = self.config.event_schedule
        if schedule is None:
            return
        for event in schedule.events_for_step(step):
            msg = apply_event(event, self.product, self.agents, self.rng)
            print(f"  {msg}")
            if event.description:
                print(f"    ({event.description})")

    def _maybe_refresh_budgets(self, step: int) -> None:
        """Replenish agent budgets on a periodic schedule.

        Models monthly income: every ``budget_refresh_interval`` steps,
        non-buyer agents get their budget reset to their original income.
        Buyers keep their post-purchase budget (they already spent on
        this product and won't buy again).
        """
        interval = self.config.budget_refresh_interval
        if interval <= 0 or step == 1:
            return
        if (step - 1) % interval != 0:
            return
        refreshed = 0
        for agent in self.agents:
            if agent.state.has_purchased:
                continue
            agent.state.resources.budget = agent.state.profile.income_amount
            agent.state.resources.spent_budget = 0.0
            refreshed += 1
        if refreshed:
            print(f"  [budget] Refreshed {refreshed} agent budgets "
                  f"(step {step}, every {interval} steps)")

    def _apply_belief_decay(self) -> None:
        """Decay sentiment and intent toward zero for non-buyers.

        Prevents saturation: without decay, a few positive WOM events
        can push sentiment to 1.0 where it stays forever, making the
        agent "unconvinceable" by negative WOM.  Decay also models the
        natural cooling-off of interest when no new stimulus arrives.

        Buyers are excluded — their sentiment reflects post-purchase
        experience and should persist.
        """
        s_decay = self.config.sentiment_decay
        i_decay = self.config.intent_decay
        for agent in self.agents:
            if agent.state.has_purchased:
                continue
            b = agent.state.beliefs
            b.overall_sentiment *= s_decay
            b.purchase_intent *= i_decay

    # ── helpers ───────────────────────────────────────────────

    def _create_agents(self) -> List[ConsumerAgent]:
        """Create agents using the factory from simulation.controller."""
        random.seed(self.config.seed)
        spec = self.config.population_spec
        tiers = _assign_tiers(self.config.n_agents, spec=spec)
        agents = []
        for i, tier in enumerate(tiers):
            agent = make_agent(f"agent_{i+1:03d}", tier, i, spec=spec)
            agents.append(agent)
        return agents

    async def _decide_with_retry(
        self, agent: ConsumerAgent, step: int, max_retries: int = 2,
    ) -> Tuple[PurchaseDecisionOutput, Event]:
        """Call agent.decide_purchase with retry on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                tier = agent.state.profile.adopter_tier
                return await agent.decide_purchase(
                    product=self.product,
                    adopted_ids=self.adopted_ids,
                    current_step=step,
                    step_duration_days=self.config.step_duration_days,
                    temperature=TIER_TEMPERATURE.get(tier, 0.5),
                    market_context=self.config.market_context,
                )
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    continue
                raise
        # Unreachable — loop either returns or raises
        raise last_exc  # type: ignore[misc]

    def _system1_event(
        self,
        agent: ConsumerAgent,
        decision: PurchaseDecisionOutput,
        filter_reason: str,
        step: int,
    ) -> Event:
        """Build an event for a System 1 (filter-only) decision.

        The event is marked so downstream analysis can distinguish filter
        rejects/defers from LLM rejects/defers.
        """
        # Apply agent state side-effects (memory, purchase_intent) without LLM
        agent._apply_decision(decision, self.product, step)
        return Event(
            event_type=EventType.PURCHASE_DECISION,
            timestamp=step,
            source=agent.agent_id,
            content={
                "product_id": self.product.product_id,
                "decision": decision.decision.value,
                "confidence": decision.confidence,
                "price_acceptable": decision.price_acceptable,
                "reasoning": decision.reasoning,
                "system1": True,
                "system1_reason": filter_reason,
            },
        )

    def _clamp_deferred(self, raw: Optional[int], step: int) -> int:
        """Clamp LLM-provided deferred_until into the valid window [step+1, step+5].

        Defends against known failure modes:
          * None -> default to next step.
          * Year-like values (e.g. 2024) -> clamp to step + 5.
          * Values <= current step -> clamp to step + 1.
        """
        lo = step + 1
        hi = step + 5
        if raw is None:
            return lo
        if raw < lo:
            print(f"    [warn] deferred_until={raw} <= step {step}; clamped to {lo}")
            return lo
        if raw > hi:
            print(f"    [warn] deferred_until={raw} > step+5; clamped to {hi}")
            return hi
        return raw

    # ── reporting ─────────────────────────────────────────────

    def _build_report(self) -> SimulationReport:
        final_tier = self.metrics.steps[-1].tier_adoption if self.metrics.steps else {}
        return SimulationReport(
            product_name=self.product.name,
            n_agents=len(self.agents),
            n_steps=self.config.n_steps,
            step_duration_days=self.config.step_duration_days,
            network_type=self.network_builder.describe(),
            wom_type=self.wom_engine.describe(),
            filter_type=self.decision_filter.describe(),
            seeding_type=self.influencer_seeding.describe(),
            belief_updater_type=self.belief_updater.describe(),
            influencer_ids=list(self.influencer_ids),
            final_adoption_rate=self.metrics.final_adoption(),
            total_sales=self.metrics.cumulative_sales(),
            total_revenue=round(self.metrics.cumulative_sales() * self.product.price, 2),
            adoption_curve=self.metrics.adoption_curve(),
            step_metrics=self.metrics.steps,
            all_events=self.all_events,
            tier_adoption=final_tier,
            system1_rejects=self.system1_rejects,
            system1_defers=self.system1_defers,
            system2_calls=self.system2_calls,
            trace_dir=str(self.trace.run_dir) if self.trace.run_dir else None,
        )

    def print_report(self, report: SimulationReport) -> None:
        """Print a formatted final report to stdout."""
        print()
        print("=" * 64)
        print(f"  SIMULATION REPORT - {report.product_name}")
        print("=" * 64)
        total_days = report.n_steps * report.step_duration_days
        print(f"  Agents        : {report.n_agents}")
        print(f"  Steps         : {report.n_steps} "
              f"({report.step_duration_days}d/step = {total_days} days total)")
        print(f"  Network       : {report.network_type}")
        print(f"  WOM engine    : {report.wom_type}")
        print(f"  Decision filter: {report.filter_type}")
        print(f"  Belief updater: {report.belief_updater_type}")
        if report.influencer_ids:
            kol_names = []
            for aid in report.influencer_ids:
                a = next((x for x in self.agents if x.agent_id == aid), None)
                if a:
                    kol_names.append(f"{a.state.profile.name} ({a.state.profile.adopter_tier})")
            print(f"  Seeding       : {report.seeding_type}")
            print(f"  KOLs ({len(report.influencer_ids)}): {', '.join(kol_names)}")
        else:
            print(f"  Seeding       : {report.seeding_type}")
        print(f"  Final adoption: {report.final_adoption_rate:.0%} "
              f"({report.total_sales}/{report.n_agents})")
        print(f"  Total revenue : ¥{report.total_revenue:,.0f}")

        total_decisions = report.system1_rejects + report.system1_defers + report.system2_calls
        if total_decisions:
            s1_share = (report.system1_rejects + report.system1_defers) / total_decisions
            print(
                f"  System 1      : {report.system1_rejects} rejects, "
                f"{report.system1_defers} defers "
                f"(saved {s1_share:.0%} of LLM calls)"
            )
            print(f"  System 2 (LLM): {report.system2_calls} calls")

        if report.trace_dir:
            print(f"  Trace dir     : {report.trace_dir}")

        self.metrics.print_adoption_curve()
        self.metrics.print_tier_breakdown()
        self.metrics.print_wom_summary()

        print()
        print("=" * 64)

        # Run automated validation checks
        from simulation.validation import validate_run
        vr = validate_run(report)
        vr.print_report()

    # ── console output ────────────────────────────────────────

    def _print_header(self) -> None:
        n = len(self.agents)
        edges = sum(len(a.state.relationships.relationships) for a in self.agents) // 2
        print()
        print("=" * 64)
        print(f"  MARKET SIMULATION - {self.product.name}")
        print(f"  {n} agents | {self.config.n_steps} steps | "
              f"{self.network_builder.describe()} ({edges} edges)")
        print("=" * 64)

    def _print_step_header(self, step: int, n_evaluators: int, reason: str) -> None:
        print()
        print("-" * 64)
        print(f"  STEP {step}/{self.config.n_steps} - evaluating {n_evaluators} agents ({reason})")
        print("-" * 64)

    def _print_step_footer(self, sm: StepMetrics, wom_gen: int, wom_rec: int) -> None:
        print()
        print(f"  Step {sm.step} result: "
              f"{sm.buyers} bought, {sm.deferrers} deferred, {sm.rejecters} rejected")
        print(f"  Cumulative adoption: {sm.adoption_rate:.0%} "
              f"({sm.cumulative_buyers}/{sm.total_agents})")
        if wom_gen:
            print(f"  WOM: {wom_gen} messages -> {wom_rec} receptions")

    def _record_empty_step(self, step: int) -> None:
        self.metrics.record_step(
            step=step,
            total_agents=len(self.agents),
            evaluators=0,
            buyers=0,
            deferrers=0,
            rejecters=0,
            cumulative_buyers=len(self.adopted_ids),
            wom_generated=0,
            wom_received=0,
            agents=self.agents,
            adopted_ids=self.adopted_ids,
        )
