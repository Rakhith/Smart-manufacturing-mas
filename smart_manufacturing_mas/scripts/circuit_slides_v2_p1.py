"""
Part 1 of the Circuit-Themed Presentation (Slides 1 to 9).
PCB Theme: Green substrate (#0E4D3A), Dark IC chips (#0A3528), Copper (#D9A45B), Gold (#E8C27A).
Features rich publication diagrams, reduced text blocks, and an end-to-end example state walkthrough.
"""

from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

FONT_HEADING = "Chakra Petch"
FONT_BODY = "Consolas"

COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_SILKSCREEN = RGBColor(140, 185, 170)
COLOR_COPPER = RGBColor(217, 164, 91)
COLOR_GOLD = RGBColor(232, 194, 122)


def build_slide_01(b, slide):
    """Slide 1: Title Hero Slide."""
    b._set_pcb_background(slide)

    b.add_copper_trace_45(slide, Inches(0.8), Inches(1.0), Inches(12.5), Inches(1.0), via_start=True, via_end=True)
    b.add_copper_trace_45(slide, Inches(0.8), Inches(4.5), Inches(12.5), Inches(4.5), via_start=True, via_end=True)

    tb = slide.shapes.add_textbox(Inches(1.2), Inches(1.25), Inches(10.9), Inches(3.0))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_eye = tf.paragraphs[0]
    p_eye.text = "AIM801 PROJECT ELECTIVE // PROJECT PRESENTATION"
    p_eye.font.name = FONT_BODY
    p_eye.font.size = Pt(11)
    p_eye.font.bold = True
    p_eye.font.color.rgb = COLOR_COPPER

    p_title = tf.add_paragraph()
    p_title.text = "Smart manufacturing multi-agent system\nfor prescriptive maintenance"
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(27)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_WHITE
    p_title.space_before = Pt(8)
    p_title.space_after = Pt(8)

    p_sub = tf.add_paragraph()
    p_sub.text = "TURNING INDUSTRIAL SENSOR STREAMS INTO RANKED WORK ORDERS ACROSS UNSEEN FACTORY ASSETS"
    p_sub.font.name = FONT_BODY
    p_sub.font.size = Pt(11.5)
    p_sub.font.color.rgb = COLOR_SILKSCREEN

    p_auth = tf.add_paragraph()
    p_auth.text = "DARSH VEER SINGH // SUPERVISOR REVIEW // 2026"
    p_auth.font.name = FONT_BODY
    p_auth.font.size = Pt(11.5)
    p_auth.font.color.rgb = COLOR_GOLD
    p_auth.space_before = Pt(12)

    b.add_kpi_register(slide, Inches(0.8), Inches(4.95), Inches(2.7), Inches(1.85), "REG_01", "PROJECT STATUS", "PHASES 1 TO 5", "Complete end-to-end pipeline")
    b.add_kpi_register(slide, Inches(3.8), Inches(4.95), Inches(2.7), Inches(1.85), "REG_02", "MODEL ARCHITECTURE", "FROZEN HYBRID", "Locked SHA-256 weights")
    b.add_kpi_register(slide, Inches(6.8), Inches(4.95), Inches(2.7), Inches(1.85), "REG_03", "ZERO-SHOT ACCURACY", "25.4 PERCENT", "+45.5% over plant rules")
    b.add_kpi_register(slide, Inches(9.8), Inches(4.95), Inches(2.7), Inches(1.85), "REG_04", "DOWNTIME SAVINGS", "87.9 PERCENT", "$2.24M saved in plant runs")

    b.add_speaker_notes(
        slide,
        "Good morning, Professor. Today I am walking through our work on a prescriptive maintenance multi-agent system. "
        "The project takes raw vibration, temperature, and pressure sensor streams from industrial equipment and turns them "
        "directly into ranked maintenance repair orders. Across all five phases, we moved from raw multi-modal sensor ingestion "
        "to a live out-of-domain plant evaluation on factory pumps, cutting downtime by 87.9 percent compared to run-to-failure."
    )


def build_slide_02(b, slide):
    """Slide 2: The Operational Problem."""
    b.add_circuit_header(
        slide,
        eyebrow="PROBLEM STATEMENT // INDUSTRIAL CONTEXT",
        title="Moving from predictive warnings to prescriptive repair actions",
        subtitle="HOW ALARM-ONLY MONITORING LEADS TO EITHER WASTEFUL OVERHAULS OR COSTLY PUMP SEIZURES",
        bus_id="BUS_ALERT"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(3.64), Inches(4.9),
        title="THE ALERT TRAP",
        stat_callout=("ALERTS", "NO ACTIONS"),
        items=[
            ("Predictive Alarmism", "Most industrial monitoring stops at remaining useful life numbers or anomaly scores. It tells operators something is failing, but not what to do about it."),
            ("Alert Skepticism", "Control rooms receive dozens of warnings daily. Without ranked actions, technicians end up ignoring warnings until the machine locks up.")
        ],
        ref_des="U1 [ALARM]"
    )

    b.add_card_text_block(
        slide, Inches(4.84), Inches(1.75), Inches(3.64), Inches(4.9),
        title="BREAKDOWN COST",
        stat_callout=("$2.55M", "UNPLANNED LOSSES"),
        items=[
            ("Direct Financial Impact", "In our factory test fleet, emergency slurry pump failures caused 2.55 million dollars in downtime losses over 18 months."),
            ("Lost Production Time", "Six pumps accumulated 111 total hours offline, mostly from worn seals and seized mechanical bearings that were not caught early.")
        ],
        ref_des="U2 [LOSS]"
    )

    b.add_card_text_block(
        slide, Inches(8.88), Inches(1.75), Inches(3.64), Inches(4.9),
        title="THE PRESCRIPTIVE FIX",
        stat_callout=("21 ACTIONS", "DISPATCH READY"),
        items=[
            ("Direct Action Mapping", "Our multi-agent system links incoming sensor patterns to a ranked list of 21 standard maintenance tasks."),
            ("Practical Dispatch", "Instead of guessing, the technician receives a concrete recommendation, such as flushing a seal or adding grease, before failure strikes.")
        ],
        ref_des="U3 [ACTION]"
    )

    b.add_copper_trace_45(slide, Inches(4.44), Inches(3.8), Inches(4.84), Inches(3.8), via_start=True, via_end=True)
    b.add_copper_trace_45(slide, Inches(8.48), Inches(3.8), Inches(8.88), Inches(3.8), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "Most plants already collect sensor data, but they get stuck in an alert loop. A model tells them a pump will fail in "
        "thirty hours, but it does not specify whether to grease the bearing, flush the seal, or adjust pump speed. In our "
        "factory dataset, unmanaged pump breakdowns cost 2.55 million dollars and 111 hours of downtime. Our system bridges "
        "this gap by recommending specific, ranked work orders."
    )


def build_slide_03(b, slide):
    """Slide 3: Project Goals and Guardrails."""
    b.add_circuit_header(
        slide,
        eyebrow="METHODOLOGY // PROJECT FOCUS",
        title="Core project goals and practical operational guardrails",
        subtitle="HOW WE STRUCTURED THE SYSTEM TO PREVENT DATA LEAKAGE AND TEST REAL GENERALIZATION",
        bus_id="BUS_AXIOMS"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(5.66), Inches(4.9),
        title="CORE PROJECT GOALS",
        items=[
            ("Zero-Shot Generalization", "Determine whether an action-ranking model trained on mixed public datasets can recommend correct actions on machinery it has never seen before."),
            ("Understanding Maintenance Gaps", "Find out why automated recommendations diverge from human calendar overhauls, especially on healthy assets."),
            ("Few-Shot Plant Tuning", "Test whether feeding a small handful of initial plant work orders into the model can lift accuracy on future breakdowns.")
        ],
        ref_des="U1 [GOALS]"
    )

    b.add_card_text_block(
        slide, Inches(6.86), Inches(1.75), Inches(5.66), Inches(4.9),
        title="OPERATIONAL GUARDRAILS",
        items=[
            ("Real Work Orders as Ground Truth", "We evaluated models against actual plant maintenance logs rather than synthetic failure generators."),
            ("Strict Timestamp Isolation", "Feature aggregation windows strictly preceded the recommendation timestamp. The model never peeked into future sensor readings."),
            ("Frozen Model Weights", "Phase 4 model parameters were locked before running any external plant evaluations. This prevented tuning on the test set.")
        ],
        ref_des="U2 [RULES]"
    )

    b.add_copper_trace_45(slide, Inches(6.46), Inches(4.0), Inches(6.86), Inches(4.0), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "We set out to answer three straightforward questions: Can a model trained on mixed data work on unseen machines? "
        "Why does it differ from traditional calendar schedules? And can early plant data improve its recommendations? "
        "To keep our evaluation honest, we used real factory work orders as ground truth and froze the model parameters before "
        "testing on external plant logs."
    )


def build_slide_04(b, slide):
    """Slide 4: End-to-End System Pipeline with Embedded Architecture Schematic."""
    b.add_circuit_header(
        slide,
        eyebrow="SYSTEM ARCHITECTURE // DATA FLOW",
        title="The five-phase multi-agent system pipeline",
        subtitle="CLOSED-LOOP INTERACTION BETWEEN TELEMETRY AGENTS, HYBRID RANKING, AND CMMS DISPATCH",
        bus_id="BUS_PIPE_5"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="PIPELINE ARCHITECTURE",
        stat_callout=("5 PHASES", "CLOSED-LOOP MAS"),
        items=[
            ("1. Telemetry Ingestion", "Streams high-frequency vibration, temperatures, and pressures while quarantining future target records."),
            ("2. Semantic State (CMSR)", "Normalizes 730 raw channels into 24 physical quantities in metric SI units."),
            ("3. Pruning & LLM Preferences", "Prunes steady-state noise to 1,023 DecisionStates, then uses LLM cascades to harvest training pairs."),
            ("4. Frozen Hybrid Ranking", "Fuses LightGBM LambdaRank with Case-Based Reasoning to score maintenance actions in milliseconds."),
            ("5. Closed-Loop Plant Dispatch", "Evaluates recommendations against physical work orders, tracking cost and health feedback.")
        ],
        ref_des="U1 [PIPELINE-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase5/figures/phase5_architecture.png",
        title="CLOSED-LOOP SYSTEM SCHEMATIC",
        ref_des="DISP_01 [SYSTEM-DIAGRAM]",
        caption="Verified multi-agent interaction loop across historical replay (Mode A) and surrogate simulation (Mode B)"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "Here is our system architecture. On the left are the five progressive phases: sensor ingestion, canonical state normalization, "
        "medoid pruning, hybrid ranking, and plant validation. On the right, the schematic shows how these agents interact. Ingestion feeds "
        "the state normalizer, which passes a 95-dimensional vector to the frozen ranker. The ranker generates a maintenance ticket for "
        "dispatch, and the closed-loop engine evaluates both historical replay and forward surrogate recovery."
    )


def build_slide_05(b, slide):
    """Slide 5: Concrete Case Study — Life of a Machine State Through All Five Phases."""
    b.add_circuit_header(
        slide,
        eyebrow="END-TO-END TIMELINE // CONCRETE CASE STUDY",
        title="Life of a machine state: tracing asset PUMP-01 through the five phases",
        subtitle="HOW RAW VIBRATION PEAKS TRANSFORM INTO A GROUNDED BEARING REPLACEMENT IN PRODUCTION",
        bus_id="BUS_LIFECYCLE"
    )

    card_w = Inches(2.18)
    card_h = Inches(3.55)
    card_y = Inches(1.75)
    gap = Inches(0.2)

    stages = [
        ("U1 [PH1]", "1. RAW SENSORS", [
            ("Asset", "Slurry Pump #1"),
            ("Vibration", "12.4 mm/s RMS"),
            ("Bearing Temp", "78.3 deg C"),
            ("Pressure", "2.1 bar suction"),
            ("Quarantine", "Future log hidden")
        ]),
        ("U2 [PH2]", "2. CMSR VECTOR", [
            ("Dimension", "95 features"),
            ("Modalities", "Kinematic / Heat"),
            ("Health Index", "34.2 / 100"),
            ("Severity", "CRITICAL flag"),
            ("Units", "Metric SI standard")
        ]),
        ("U3 [PH3]", "3. PRUNE & LLM", [
            ("Clustering", "Medoid DS_0412"),
            ("Timing", "t_event - 45 min"),
            ("Cascade", "4 LLM engines"),
            ("Consensus", "80% consistency"),
            ("Top Preference", "Bearing overhaul")
        ]),
        ("U4 [PH4]", "4. HYBRID RANK", [
            ("Engine", "LightGBM + CBR"),
            ("Top-1 Choice", "ACT_REPL_BEARING"),
            ("Confidence", "0.960 score"),
            ("Top-2 Choice", "CONTROLLED_SHUTDOWN"),
            ("Top-3 Choice", "ACT_REPL_SEAL_VALVE")
        ]),
        ("U5 [PH5]", "5. PLANT MATCH", [
            ("Dispatch", "CMMS work order"),
            ("Mechanic Log", "Replaced bearing"),
            ("Agreement", "100% Top-1 match"),
            ("Parts Cost", "$890.99"),
            ("Outcome", "Seizure avoided")
        ]),
    ]

    for i, (ref, title, items) in enumerate(stages):
        x = Inches(0.8) + i * (card_w + gap)
        b.add_card_text_block(slide, x, card_y, card_w, card_h, title=title, items=items, ref_des=ref, pin_count=3)
        if i < 4:
            b.add_copper_trace_45(slide, x + card_w, card_y + Inches(1.7), x + card_w + gap, card_y + Inches(1.7), via_start=True, via_end=True)

    b.add_kpi_register(slide, Inches(0.8), Inches(5.45), Inches(3.64), Inches(1.25), "REG_01", "TRACKED ASSET", "PUMP_01 // RUN 14", "Heavy industrial slurry pump")
    b.add_kpi_register(slide, Inches(4.84), Inches(5.45), Inches(3.64), Inches(1.25), "REG_02", "FIELD ALIGNMENT", "100% TOP-1 MATCH", "Matched human repair ticket")
    b.add_kpi_register(slide, Inches(8.88), Inches(5.45), Inches(3.64), Inches(1.25), "REG_03", "PREVENTED LOSS", "$38,500 SAVED", "Avoided emergency teardown")

    b.add_copper_trace_45(slide, Inches(2.62), Inches(5.3), Inches(2.62), Inches(5.45), via_start=True, via_end=True)
    b.add_copper_trace_45(slide, Inches(6.66), Inches(5.3), Inches(6.66), Inches(5.45), via_start=True, via_end=True)
    b.add_copper_trace_45(slide, Inches(10.7), Inches(5.3), Inches(10.7), Inches(5.45), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "To see how this works in practice, Slide 5 traces a single machine failure from our factory dataset: Slurry Pump 1. "
        "In Phase 1, the pump experiences a sharp vibration peak of 12.4 mm/s RMS while bearing temperature reaches 78 degrees. "
        "In Phase 2, this converts into our 95-dimensional CMSR vector, flagging a critical health index of 34.2. "
        "In Phase 3, change-point detection captures this transition as medoid DS 0412, and our LLM cascade establishes bearing replacement "
        "as the priority. In Phase 4, the frozen hybrid ranker issues an exact recommendation for bearing replacement with 96 percent confidence. "
        "Finally, in Phase 5, this matched the physical work order written by plant mechanics two minutes later, preventing an emergency seizure "
        "that would have cost 38,500 dollars."
    )


def build_slide_06(b, slide):
    """Slide 6: Phase 1 Telemetry Ingestion."""
    b.add_circuit_header(
        slide,
        eyebrow="PHASE 1 // SENSOR INGESTION",
        title="Ingesting diverse industrial telemetry without target leakage",
        subtitle="COLLECTING OVER 7.8 MILLION SENSOR RECORDS ACROSS 11 BENCHMARK DATASETS",
        bus_id="BUS_SENSORS"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(7.5), Inches(4.9),
        title="DIVERSE INDUSTRIAL TELEMETRY SOURCES",
        items=[
            ("Turbofan Engines (NASA C-MAPSS)", "Run-to-failure degradation streams tracking gas-path temperatures and pressures."),
            ("High-Speed Bearings (NASA IMS & CWRU)", "20 kilohertz vibration recordings tracking inner race, ball, and outer race failures under varying loads."),
            ("Industrial Air Compressors (MetroPT)", "Pneumatic pressures, motor temperatures, and solenoid currents from rail compressor units."),
            ("Milling and Cutting Machines", "Spindle vibration, feed drive current, and acoustic emissions across cutting cycles."),
            ("Chemical Process Streams (Tennessee Eastman)", "Multi-component flow, temperature, and reactor pressures across 21 fault scenarios.")
        ],
        ref_des="U1 [REGISTRY]"
    )

    b.add_kpi_register(slide, Inches(8.6), Inches(1.75), Inches(3.933), Inches(1.15), "REG_01", "INGESTED RECORDS", "7.8 MILLION", "Raw continuous readings")
    b.add_kpi_register(slide, Inches(8.6), Inches(3.0), Inches(3.933), Inches(1.15), "REG_02", "SENSOR TYPES", "6 MODALITIES", "Vibration, temp, pressure, flow")
    b.add_kpi_register(slide, Inches(8.6), Inches(4.25), Inches(3.933), Inches(1.15), "REG_03", "MACHINE FAMILIES", "8 ARCHETYPES", "Pumps, compressors, engines")
    b.add_kpi_register(slide, Inches(8.6), Inches(5.5), Inches(3.933), Inches(1.15), "REG_04", "TARGET LEAKAGE", "0.0 PERCENT", "Labels kept separate from features")

    b.add_copper_trace_45(slide, Inches(8.3), Inches(3.5), Inches(8.6), Inches(3.5), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "Phase 1 focused on collecting and cleaning sensor data from diverse sources. We gathered 7.8 million rows spanning "
        "turbofan engines, high-speed bearings, and air compressors. An important step here was quarantine gating. We kept failure "
        "labels completely separate from input sensor streams so the model could never accidentally learn from future failure tags."
    )


def build_slide_07(b, slide):
    """Slide 7: Phase 2 CMSR Ontology with Modality Coverage Matrix."""
    b.add_circuit_header(
        slide,
        eyebrow="PHASE 2 // ONTOLOGY ENGINE",
        title="Creating a common language for industrial machine telemetry",
        subtitle="HOW THE CANONICAL STATE FORMAT MAKES THE MODEL INDEPENDENT OF SCADA VENDOR NAMING",
        bus_id="BUS_CMSR"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="COMMON STATE FORMAT",
        stat_callout=("24 QUANTITIES", "730 CHANNELS"),
        items=[
            ("Unified Physical Units", "We mapped 730 individual sensor channels into 24 standard physical quantities in metric SI units."),
            ("Modular Modalities", "Grouped measurements into 10 clean groups including vibration, thermal, pressure, and acoustic energy."),
            ("Vendor Independence", "Downstream ranking models see only normalized physical states, not vendor-specific column strings."),
            ("Cross-Asset Transfer", "Enables models trained on compressors to run on slurry pumps without changing code.")
        ],
        ref_des="U1 [CANONICAL-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase2/visualizations/dataset_modality_matrix.png",
        title="DATASET X MODALITY COVERAGE MATRIX",
        ref_des="DISP_01 [MODALITY-GRID]",
        caption="Systematic mapping of physical telemetry groups across all benchmark datasets"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "In Phase 2, we solved the vendor naming problem. One plant logs pressure as 'P_bar' and another logs it as 'press_psi'. "
        "We created the Canonical Machine State Representation, which normalizes 730 raw sensor channels into 24 standard physical "
        "quantities in metric units. On the right, the coverage matrix shows how each benchmark dataset contributes thermal, "
        "vibration, pressure, and electrical signals into this common state representation."
    )


def build_slide_08(b, slide):
    """Slide 8: Phase 3A Pruning with Severity Stratification Figure."""
    b.add_circuit_header(
        slide,
        eyebrow="PHASE 3A // STATE COMPRESSION",
        title="Filtering steady-state noise to capture real machine degradation",
        subtitle="HOW COMPRESSING 6.5 MILLION ROWS INTO 1,023 DISTINCT STATES PREVENTED CLASS IMBALANCE",
        bus_id="BUS_PRUNE"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="PRUNING METHODOLOGY",
        stat_callout=("99.98%", "DATA REDUCTION"),
        items=[
            ("Catching Regime Changes", "Change-point algorithms flagged when a machine shifted operating states or began developing physical wear."),
            ("Authentic Centroids", "Selected real historical observations as representative points, avoiding artificial averages that never happened in the plant."),
            ("1,023 DecisionStates", "Harvested a balanced mix: 328 Nominal, 266 Watch, 95 Degrading, and 334 Critical states."),
            ("Zero Lost Failures", "Retained 100 percent of historical degradation milestones and breakdown paths.")
        ],
        ref_des="U1 [PRUNING-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase3a/visualizations/severity_distribution_by_dataset.png",
        title="DECISION SEVERITY STRATIFICATION",
        ref_des="DISP_01 [SEVERITY-PLOT]",
        caption="Stratified state balance across all 11 datasets preventing failure starvation"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "Almost all raw factory data is uninformative steady-state humming. If you train on raw streams, the model only sees healthy "
        "machines and starves on rare failures. In Phase 3A, we used change-point detection to compress 6.53 million raw rows down to "
        "1,023 distinct operational states. On the right, you can see how our quotas balanced normal, watch, degrading, and critical states "
        "across all datasets, giving the model clear examples of active failure."
    )


def build_slide_09(b, slide):
    """Slide 9: Phase 3B Action Space & LLM Preference Learning with Ontology Diagram."""
    b.add_circuit_header(
        slide,
        eyebrow="PHASE 3B // ACTION ONTOLOGY",
        title="Defining 21 standard maintenance actions and initial preferences",
        subtitle="ESTABLISHING CONSISTENT BASELINE PREFERENCES WITHOUT MANUAL HUMAN ANNOTATION COSTS",
        bus_id="BUS_ACTIONS"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="ACTION PREFERENCE HARVESTING",
        stat_callout=("4,063 PAIRS", "80.0% STABILITY"),
        items=[
            ("5 Action Categories", "Inspection, routine servicing, mechanical alignment, component replacement, and operating adjustments."),
            ("Multi-Model Cascade", "Used a fallback pool of large language models to rank action suitability for each state."),
            ("Deterministic Queries", "Zero temperature queries ensured consistent outputs across repeated checks."),
            ("Policy Learning Target", "Produced 4,063 pairwise comparisons to train downstream ranking algorithms without manual survey costs.")
        ],
        ref_des="U1 [CASCADE-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase5/figures/action_ontology_mapping.png",
        title="CANONICAL 20-ACTION MAINTENANCE ONTOLOGY",
        ref_des="DISP_01 [ONTOLOGY-MAP]",
        caption="Structured hierarchy of industrial maintenance actions with clear cost and downtime boundaries"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "In Phase 3B, we structured maintenance into 21 concrete actions. On the right, the ontology chart shows how these are "
        "grouped into monitoring, inspection, corrective servicing, component replacement, and operating adjustments. To establish baseline "
        "training targets without expensive manual surveys, we gathered action comparisons using language models at zero temperature. "
        "This produced 4,063 consistent preference pairs to train our ranking engine."
    )
