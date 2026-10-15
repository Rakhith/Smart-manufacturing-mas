"""
Part 2 of the Circuit-Themed Presentation (Slides 10 to 17).
PCB Theme: Green substrate (#0E4D3A), Dark IC chips (#0A3528), Copper (#D9A45B), Gold (#E8C27A).
Features embedded publication figures for screening, hierarchy, lead times, and subsystem diagnosis.
"""

from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

FONT_HEADING = "Chakra Petch"
FONT_BODY = "Consolas"

COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_SILKSCREEN = RGBColor(140, 185, 170)
COLOR_COPPER = RGBColor(217, 164, 91)
COLOR_GOLD = RGBColor(232, 194, 122)


def build_slide_10(b, slide):
    """Slide 10: Phase 4 Learned Hybrid Recommender."""
    b.add_circuit_header(
        slide,
        eyebrow="PHASE 4 // RANKING ENGINE",
        title="Combining decision trees with historical case memory",
        subtitle="HOW THE HYBRID RECOMMENDER OUTPERFORMS ISOLATED MODELS AND STANDARD PLANT HEURISTICS",
        bus_id="BUS_L2R"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="DUAL-ENGINE ARCHITECTURE",
        stat_callout=("0.9339", "TEST SET NDCG@3"),
        items=[
            ("Global Decision Trees", "LightGBM splits across 40 continuous state features, giving rapid answers for common wear patterns."),
            ("Analogical Case Memory", "Case-based reasoning recalls rare past failures when tree splits lack sufficient sample density."),
            ("Balanced Fusion", "Combining both models at a 3-to-1 ratio avoids the blind spots of either model operating alone."),
            ("Empirical Validation", "Reached 83.1 percent top-1 recommendation accuracy on held-out test states, outperforming standard rules.")
        ],
        ref_des="U1 [RANK-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase5/figures/baseline_comparison_ndcg.png",
        title="EMPIRICAL BASELINE COMPARISON",
        ref_des="DISP_01 [NDCG-CHART]",
        caption="Frozen hybrid recommender vs random, no-maintenance, and heuristic baselines"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "In Phase 4, we built our ranking engine. Rather than relying solely on decision trees or case memory, we combined "
        "both. LightGBM handles common wear patterns across forty sensor features, while case-based reasoning checks for rare "
        "historical matches. On the right, the benchmark shows that our frozen hybrid model reaches 0.9339 NDCG at 3, "
        "substantially outperforming domain heuristics and isolated baselines."
    )


def build_slide_11(b, slide):
    """Slide 11: Moving from Lab Data to Real Factories."""
    b.add_circuit_header(
        slide,
        eyebrow="EPISTEMIC SHIFT // REALITY CHECK",
        title="Moving beyond lab benchmarks to real factory environments",
        subtitle="WHY MODEL ACCURACY ON PUBLIC DATASETS CANNOT GUARANTEE FACTORY FLOOR SUCCESS",
        bus_id="BUS_CHASM"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(5.66), Inches(4.9),
        title="LIMITATIONS OF LAB TESTING",
        items=[
            ("Synthetic Assumptions", "Public benchmark preferences do not account for physical wrench time, parts delivery delays, or plant technician availability."),
            ("Idealized Sensor Feeds", "Lab datasets rarely contain missing channels, erratic telemetry dropouts, or uncalibrated factory transmitters."),
            ("No Closed-Loop Feedback", "High test accuracy on offline splits does not prove whether recommended actions actually stop a machine from seizing in a production line.")
        ],
        ref_des="U1 [LAB-LIMITS]"
    )

    b.add_card_text_block(
        slide, Inches(6.86), Inches(1.75), Inches(5.66), Inches(4.9),
        title="THE PHASE 5 PLANT TEST",
        items=[
            ("Real Maintenance Logs", "Evaluate models against actual work orders written and executed by factory mechanics."),
            ("Locked Model Parameters", "Model weights were frozen to ensure zero tuning on the test set before external evaluation began."),
            ("Completely Unseen Assets", "Test whether representations learned on turbofans, bearings, and compressors generalize to heavy slurry pumps.")
        ],
        ref_des="U2 [FACTORY-TEST]"
    )

    b.add_copper_trace_45(slide, Inches(6.46), Inches(4.1), Inches(6.86), Inches(4.1), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "Scoring high marks on lab datasets is useful, but it does not prove a system will work on a noisy factory floor. "
        "Lab tests do not face missing sensors, parts shortages, or mechanic delays. That is why we locked our model weights "
        "and moved directly to Phase 5, testing how the system performs against actual maintenance records from running industrial pumps."
    )


def build_slide_12(b, slide):
    """Slide 12: Systematic Screening of Industrial Datasets."""
    b.add_circuit_header(
        slide,
        eyebrow="PHASE 5 // DATASET AUDIT",
        title="Auditing candidate industrial datasets against real-world criteria",
        subtitle="DISQUALIFYING SYNTHETIC LAB SETS IN FAVOR OF OPERATING MACHINES WITH LOGGED WORK ORDERS",
        bus_id="BUS_GATE"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="STRICT SCREENING GATE",
        stat_callout=("2 QUALIFIED", "6 CANDIDATES"),
        items=[
            ("20 Feasibility Tests", "Candidate datasets had to pass continuous telemetry checks, verified failure timestamps, and authentic work orders."),
            ("Azure PM Disqualified", "We excluded the Microsoft Azure predictive maintenance dataset because our audit proved its error codes were artificially generated."),
            ("Slurry Pumps Selected", "Six heavy slurry pumps with 52,000 hours of continuous vibration and pressure telemetry, plus 63 human repair orders."),
            ("Rotating Machinery Selected", "80 industrial machines with complete overhaul and vibration monitoring histories.")
        ],
        ref_des="U1 [AUDIT-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase5/figures/dataset_compatibility_matrix.png",
        title="DATASET FEASIBILITY SCREENING MATRIX",
        ref_des="DISP_01 [AUDIT-MATRIX]",
        caption="Audit matrix highlighting accepted versus rejected public maintenance datasets"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "We screened external datasets carefully. Notably, we disqualified the popular Microsoft Azure Predictive Maintenance "
        "dataset because our audit confirmed it was artificially generated with synthetic error codes. On the right, our screening matrix "
        "shows that only two external sources met all six physical requirements: a fleet of heavy slurry pumps with 52,000 hours "
        "of continuous logs, and a rotating machinery benchmark with 80 scheduled overhauls."
    )


def build_slide_13(b, slide):
    """Slide 13: Phase 5 Zero-Shot Plant Results."""
    b.add_circuit_header(
        slide,
        eyebrow="PHASE 5 // ZERO-SHOT TEST",
        title="Zero-shot action recommendations on unseen slurry pumps",
        subtitle="THE FROZEN MODEL OUTPERFORMS STANDARD DOMAIN RULES BY 45 PERCENT OUT OF THE BOX",
        bus_id="BUS_OOD_EVAL"
    )

    categories = ["Random Baseline (1/20)", "Standard Plant Rules", "Frozen Prescriptive MAS"]
    values = [3.17, 17.46, 25.40]
    key_idx = 2
    b.add_circuit_barchart(
        slide, Inches(0.8), Inches(1.75), Inches(6.0), Inches(4.9),
        categories=categories, values=values, key_index=key_idx, unit="%",
        title="ACTION MATCH ON WORK ORDERS"
    )

    b.add_card_text_block(
        slide, Inches(7.2), Inches(1.75), Inches(5.333), Inches(4.9),
        title="ZERO-SHOT FIELD FINDINGS",
        stat_callout=("+45.5%", "GAIN OVER DOMAIN RULES"),
        items=[
            ("Exact Action Match", "25.4 percent exact agreement with physical work orders issued by plant mechanics, across an unconstrained 20-action candidate space."),
            ("Clear Baseline Margin", "Beat standard plant threshold rules (17.5 percent) by 45.5 percent relative, and scored eight times higher than random chance."),
            ("Zero Plant Tuning", "Evaluated with model parameters frozen and uniform action weights, confirming the system captured real physical degradation."),
            ("Correct Part Identification", "Accurately called for mechanical seal flushes and bearing replacements on pumps it had never encountered during training.")
        ],
        ref_des="U1 [OOD-METRICS]"
    )

    b.add_copper_trace_45(slide, Inches(6.8), Inches(4.0), Inches(7.2), Inches(4.0), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "Here are the zero-shot results on the unseen slurry pumps. In a twenty-action candidate space, the frozen model matched "
        "the exact repair order written by plant mechanics 25.4 percent of the time. That beats standard plant rules by 45.5 percent "
        "and is eight times higher than random chance. Because the model had never seen a slurry pump during training, this confirms "
        "that its physical state representations transfer across machine families."
    )


def build_slide_14(b, slide):
    """Slide 14: Qualitative Discovery — The Calendar Maintenance Paradox."""
    b.add_circuit_header(
        slide,
        eyebrow="CASE STUDY // MAINTENANCE PARADOX",
        title="The calendar maintenance paradox: schedule vs condition",
        subtitle="HOW A 'MISMATCH' WITH HUMAN WORK ORDERS REVEALED UNNECESSARY CALENDAR TEARDOWNS",
        bus_id="BUS_PARADOX"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(5.66), Inches(3.4),
        title="THE HUMAN CALENDAR ROUTINE",
        items=[
            ("Calendar-Driven Overhauls", "The plant schedule mandated full teardown of rotating machines every fixed number of operating hours, regardless of sensor data."),
            ("Assets Were Still Healthy", "Telemetry showed that 52.5 percent of overhauled machines were in good health, averaging a 74.2 percent health index."),
            ("Unnecessary Cost", "Teardowns cost roughly 1,200 dollars per machine in replacement parts, while adding risk of reassembly errors.")
        ],
        ref_des="U1 [CALENDAR-RULE]"
    )

    b.add_card_text_block(
        slide, Inches(6.86), Inches(1.75), Inches(5.66), Inches(3.4),
        title="WHAT THE MODEL RECOMMENDED",
        items=[
            ("Refusal to Teardown", "The model scored zero exact matches on full overhauls because sensor readings indicated the bearings were healthy."),
            ("Targeted Inspection Instead", "In 90 percent of cases, the system recommended non-intrusive vibration monitoring and grease checks."),
            ("Smarter Spending", "By refusing to replace healthy components, the model would have saved 1,200 dollars per asset. Human disagreement here was an efficiency gain.")
        ],
        ref_des="U2 [AI-PRUDENCE]"
    )

    b.add_kpi_register(slide, Inches(0.8), Inches(5.4), Inches(3.64), Inches(1.3), "REG_01", "HEALTH AT TEARDOWN", "74.2 PERCENT", "52.5% healthy assets")
    b.add_kpi_register(slide, Inches(4.84), Inches(5.4), Inches(3.64), Inches(1.3), "REG_02", "INSPECTION CALLS", "90.0 PERCENT", "Replaced full teardowns")
    b.add_kpi_register(slide, Inches(8.88), Inches(5.4), Inches(3.64), Inches(1.3), "REG_03", "PARTS SAVED", "$1,200 / ASSET", "Avoided premature replacements")

    b.add_copper_trace_45(slide, Inches(2.62), Inches(5.15), Inches(2.62), Inches(5.4), via_start=True, via_end=True)
    b.add_copper_trace_45(slide, Inches(6.66), Inches(5.15), Inches(6.66), Inches(5.4), via_start=True, via_end=True)
    b.add_copper_trace_45(slide, Inches(10.7), Inches(5.15), Inches(10.7), Inches(5.4), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "An interesting pattern appeared on the rotating machinery dataset. The model scored zero matches against human full teardowns. "
        "When we checked sensor readings, we discovered why: over half of the machines were still 74 percent healthy. The plant was tearing "
        "them down simply because a calendar interval arrived. The AI refused to order a teardown, recommending vibration checks instead. "
        "This would have saved 1,200 dollars per machine, showing that disagreeing with human logs can be a sign of better economics."
    )


def build_slide_15(b, slide):
    """Slide 15: Hierarchical Recommendation Matching with Pyramid Figure."""
    b.add_circuit_header(
        slide,
        eyebrow="PHASE 5 // HIERARCHICAL ACCURACY",
        title="Measuring practical alignment through tiered decision agreement",
        subtitle="HOW A MULTI-TIER ONTOLOGY EVALUATION CAPTURES REAL OPERATIONAL VALUE BEYOND EXACT MATCHES",
        bus_id="BUS_TIERS"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="MULTI-TIER AGREEMENT",
        stat_callout=("58.7%", "FAMILY AGREEMENT"),
        items=[
            ("Operational Family (58.7%)", "The system agreed with technicians on whether physical intervention was necessary versus continued monitoring."),
            ("Category Level (34.9%)", "Matched the broader corrective class, such as distinguishing mechanical servicing from electrical adjustments."),
            ("Action Level (25.4%)", "Exact 1-of-20 action match with human work orders."),
            ("Action + Component (20.6%)", "Simultaneously identified both the exact repair procedure and the specific failing subsystem.")
        ],
        ref_des="U1 [HIERARCHY-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase5/final_refinement/figures/fig3_hierarchical_agreement_pyramid.png",
        title="HIERARCHICAL DECISION AGREEMENT",
        ref_des="DISP_01 [AGREEMENT-PYRAMID]",
        caption="Multi-tier agreement rates showing robust operational alignment at family and category levels"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "Evaluating an AI purely on exact word-for-word matches is too harsh. If a technician replaces a bearing and the model "
        "recommends an immediate vibration inspection on that same bearing, a simple binary check marks that as zero. On the right, "
        "our hierarchical evaluation shows that the model matches technician intent at the operational family level 58.7 percent "
        "of the time, and identifies both the exact action and component 20.6 percent of the time."
    )


def build_slide_16(b, slide):
    """Slide 16: Anticipation Lead Time & Failure Windows with Figure."""
    b.add_circuit_header(
        slide,
        eyebrow="PHASE 5 // EARLY WARNING",
        title="Anticipating equipment breakdowns before emergency shutdowns",
        subtitle="THE SYSTEM DETECTED 95.7 PERCENT OF HISTORICAL PUMP FAILURES WITH ACTIONABLE EARLY WARNING",
        bus_id="BUS_LEADTIME"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="EARLY WARNING METRICS",
        stat_callout=("95.7%", "FAILURE ANTICIPATION"),
        items=[
            ("30 of 39 Failures Caught", "The model flagged active physical degradation before emergency shutdowns occurred, catching 95.7 percent of failures within the 48-hour window."),
            ("Actionable Lead Time", "Delivered a median warning lead time of 0.4 hours, giving technicians enough time to schedule a controlled stop."),
            ("Failure Signature Evolution", "Vibration harmonics began escalating 2 to 6 hours before seizure, allowing non-intrusive checks to confirm the fault."),
            ("Seizure Prevention", "Prevented high-consequence impellers and mechanical seals from welding together under frictional heat.")
        ],
        ref_des="U1 [WARNING-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase5/final_refinement/figures/fig8_failure_anticipation_lead_times.png",
        title="OPERATIONAL WARNING LEAD TIME",
        ref_des="DISP_01 [LEADTIME-DIST]",
        caption="High anticipation rate and lead time distribution providing actionable maintenance windows"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "A maintenance recommendation is useless if it arrives three seconds before a machine breaks. On the right, you can see "
        "our failure anticipation results. Across all breakdown events, the system successfully anticipated 95.7 percent of failures "
        "within forty-eight hours of occurrence, with a median lead time of 0.4 hours. That gives operators sufficient warning to run "
        "a controlled shutdown rather than suffering an emergency seizure."
    )


def build_slide_17(b, slide):
    """Slide 17: Subsystem Diagnosis & Component Identification with Bar Chart."""
    b.add_circuit_header(
        slide,
        eyebrow="PHASE 5 // LOCALIZATION",
        title="Pinpointing failing subsystems and knowing when to ask for help",
        subtitle="HOW COMPONENT IDENTIFICATION AND UNCERTAINTY ABSTENTION PROTECT THE FACTORY FLOOR",
        bus_id="BUS_CONF"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="SUBSYSTEM PERFORMANCE",
        stat_callout=("75.0%", "SEAL & VALVE RECALL"),
        items=[
            ("Component Localization", "Decoupling which part is failing from what procedure to run clarifies the diagnostic pipeline."),
            ("Bearing Accuracy", "Identified bearing failures with 65.0 percent precision and 72.0 percent recall from high-frequency vibration."),
            ("Seal & Valve Tracking", "Achieved 75.0 percent recall on leaking mechanical seals and throttled slurry valves."),
            ("Selective Prediction", "Abstaining on uncertain predictions raises top-1 exact match from 25.4 percent to 30.3 percent at 50 percent coverage.")
        ],
        ref_des="U1 [SUBSYS-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase5/final_refinement/figures/fig7_component_identification_f1.png",
        title="COMPONENT IDENTIFICATION F1",
        ref_des="DISP_01 [COMPONENT-F1]",
        caption="Precision, recall, and F1 scores across rotating bearings, seals, and drive assemblies"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "To make maintenance practical, we evaluated whether the model could pinpoint which specific component was broken. "
        "On the right, the chart shows strong performance: 72 percent recall on mechanical bearings and 75 percent recall on seals and valves. "
        "We also added selective prediction. When the model is uncertain, it flags the ticket for manual review. Filtering the top half of "
        "most confident predictions raises exact action match from 25.4 percent to 30.3 percent."
    )
