"""
Part 3 of the Circuit-Themed Presentation (Slides 18 to 23).
PCB Theme: Green substrate (#0E4D3A), Dark IC chips (#0A3528), Copper (#D9A45B), Gold (#E8C27A).
Features embedded publication figures for adaptation curves, cost sensitivity, and comprehensive radar summary.
"""

from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

FONT_HEADING = "Chakra Petch"
FONT_BODY = "Consolas"

COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_SILKSCREEN = RGBColor(140, 185, 170)
COLOR_COPPER = RGBColor(217, 164, 91)
COLOR_GOLD = RGBColor(232, 194, 122)


def build_slide_18(b, slide):
    """Slide 18: Adapting to a New Plant with Early Records & Data Efficiency Curve."""
    b.add_circuit_header(
        slide,
        eyebrow="ADAPTATION // PLANT TRANSFER",
        title="Adapting to a new factory using its first few work orders",
        subtitle="HOW TUNING ACTION BASE RATES DOUBLED TOP-3 REPAIR RECALL ON UNTOUCHED FUTURE EVENTS",
        bus_id="BUS_ADAPT"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="ACTION PRIOR ADAPTATION",
        stat_callout=("75.0%", "TOP-3 REPAIR RECALL"),
        items=[
            ("Strict Chronological Split", "We calibrated action frequency priors using early historical events (43 work orders), then tested strictly on the remaining 20 untouched future events."),
            ("Top-3 Recall Leap", "Recall jumped from 35.0 percent to 75.0 percent, more than doubling recommendation coverage on future repairs."),
            ("Ranking Quality Lift", "Mean reciprocal rank rose from 0.329 to 0.472 (p = 0.0010 on bootstrap test), placing the correct repair action closer to the top."),
            ("Rapid Calibration", "Updating action base rates takes seconds and avoids overfitting heavy decision trees to small sample sizes.")
        ],
        ref_des="U1 [ADAPT-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase5/final_refinement/figures/fig13_data_efficiency_curve.png",
        title="FEW-SHOT ADAPTATION DATA-EFFICIENCY CURVE",
        ref_des="DISP_01 [EFFICIENCY-CURVE]",
        caption="Data efficiency curve demonstrating rapid climb in MRR and Top-1 match across training fractions"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "When bringing a system to a new factory, you do not want to retrain the entire model on just sixty work orders. "
        "That causes heavy overfitting. Instead, we calibrated action base rates using forty-three early records and tested "
        "on twenty untouched future events. On the right, the data efficiency curve shows how quickly the model climbs: "
        "with only twenty-five percent of plant records, performance exceeds the zero-shot baseline, eventually reaching seventy-five percent recall."
    )


def build_slide_19(b, slide):
    """Slide 19: Downtime Reduction and Financial Impact with Cost Sensitivity Chart."""
    b.add_circuit_header(
        slide,
        eyebrow="BUSINESS IMPACT // COST SIMULATION",
        title="Reducing plant downtime and avoiding emergency repair losses",
        subtitle="HOW PROACTIVE INTERVENTIONS SAVED 2.24 MILLION DOLLARS COMPARED TO RUN-TO-FAILURE",
        bus_id="BUS_COST_DYN"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="CLOSED-LOOP PLANT ECONOMICS",
        stat_callout=("$2.24M", "NET COST SAVINGS"),
        items=[
            ("Downtime Reduction (-79.6%)", "Simulated proactive interventions cut fleet downtime from 111.4 hours down to 22.7 hours over 18 operating months."),
            ("Averting Catastrophic Seizure", "Dispatching seal replacement and grease flushes when health dropped below 60 percent reset asset health back to 92 percent."),
            ("87.9% Expense Cut", "Baseline losses dropped from 2.55 million dollars under run-to-failure down to 310 thousand dollars with prescriptive recommendations."),
            ("Fast Payback", "The entire software and hardware setup pays for itself within the first two prevented catastrophic pump seizures.")
        ],
        ref_des="U1 [ECONOMICS-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase5/final_refinement/figures/fig15_cost_sensitivity_analysis.png",
        title="OPERATIONAL COST SENSITIVITY",
        ref_des="DISP_01 [COST-PLOT]",
        caption="Operational costs across optimistic, baseline, and conservative downtime penalty regimes"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "Slide 19 translates these technical numbers into plant economics. In closed-loop simulations, catching degradation "
        "early and replacing worn seals cut fleet downtime by 79.6 percent. On the right, our cost sensitivity analysis confirms "
        "that across optimistic, baseline, and conservative financial regimes, the prescriptive AI policy consistently cuts operational "
        "losses by more than half compared to reactive repair or run-to-failure."
    )


def build_slide_20(b, slide):
    """Slide 20: Data Integrity and Automated Tests."""
    b.add_circuit_header(
        slide,
        eyebrow="AUDIT // DATA INTEGRITY",
        title="Verifying temporal causality and automated test coverage",
        subtitle="14 AUDIT CHECKS AND 21 AUTOMATED PYTEST CASES CERTIFY ZERO LOOKAHEAD LEAKAGE",
        bus_id="BUS_AUDIT"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(5.66), Inches(3.4),
        title="14-POINT CAUSAL INTEGRITY AUDIT",
        items=[
            ("Strict Timestamp Checks", "Every rolling sensor window strictly preceded the prediction timestamp. No future data entered feature inputs."),
            ("Feature Quarantine", "Failure labels and work order notes were kept isolated from model inputs."),
            ("Locked Model Hashes", "Phase 4 model weights matched their SHA-256 fingerprints before and after testing."),
            ("Separate Case Memory", "Case retrieval memory contained zero records from external plant test sets."),
            ("Audit Verdict", "All 14 causal integrity checkpoints passed with zero flags.")
        ],
        ref_des="U1 [AUDIT-GUARD]"
    )

    b.add_card_text_block(
        slide, Inches(6.86), Inches(1.75), Inches(5.66), Inches(3.4),
        title="AUTOMATED TEST SUITE",
        items=[
            ("Comprehensive Test Suite", "Unit and integration tests covering data pipelines, metrics, and models in tests/test_phase5_external_validation.py."),
            ("Metric Math Checks", "Verified calculations for exact match, mean rank, and hierarchical agreement."),
            ("Screening Gate Tests", "Verified automated disqualification of synthetic benchmarks like Azure PM."),
            ("Pytest Results", "All 21 automated test cases pass with a 100 percent success rate.")
        ],
        ref_des="U2 [TEST-SUITE]"
    )

    b.add_kpi_register(slide, Inches(0.8), Inches(5.4), Inches(3.64), Inches(1.3), "REG_01", "CAUSAL INTEGRITY", "14 / 14 PASSED", "Zero lookahead leakage")
    b.add_kpi_register(slide, Inches(4.84), Inches(5.4), Inches(3.64), Inches(1.3), "REG_02", "AUTOMATED TESTS", "21 / 21 PASSING", "100% pytest pass rate")
    b.add_kpi_register(slide, Inches(8.88), Inches(5.4), Inches(3.64), Inches(1.3), "REG_03", "MODEL HASHES", "VERIFIED MATCH", "Cryptographically frozen")

    b.add_copper_trace_45(slide, Inches(2.62), Inches(5.15), Inches(2.62), Inches(5.4), via_start=True, via_end=True)
    b.add_copper_trace_45(slide, Inches(6.66), Inches(5.15), Inches(6.66), Inches(5.4), via_start=True, via_end=True)
    b.add_copper_trace_45(slide, Inches(10.7), Inches(5.15), Inches(10.7), Inches(5.4), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "To be certain our numbers were not inflated by accidental lookahead leakage, we ran an automated fourteen-point audit. "
        "The checks verified that all input windows strictly stopped before the recommendation timestamp, case libraries were "
        "sealed against test data, and model hashes remained unchanged. Alongside that, twenty-one automated pytest test cases pass "
        "with a 100 percent success rate."
    )


def build_slide_21(b, slide):
    """Slide 21: What Is Demonstrated vs Practical Boundaries."""
    b.add_circuit_header(
        slide,
        eyebrow="EVALUATION // SCOPE BOUNDARIES",
        title="Demonstrated capabilities and practical operational boundaries",
        subtitle="CLEAR DISTINCTION BETWEEN TESTED REALITIES AND SITUATIONS REQUIRING HUMAN OVERSIGHT",
        bus_id="BUS_BOUNDS"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(5.66), Inches(4.9),
        title="WHAT WE DEMONSTRATED",
        items=[
            ("Cross-Asset Transfer", "Models trained on mixed sensor benchmarks successfully recommended actions on unseen slurry pumps, beating standard rules by 45 percent."),
            ("Prudent Asset Management", "The system correctly refused full calendar teardowns on healthy rotating equipment, recommending low-cost vibration monitoring."),
            ("Actionable Early Warning", "Anticipated 95.7 percent of pump breakdowns within a 48-hour window with median lead times between 0.4 and 6.0 hours."),
            ("Quick Plant Tuning", "Action frequency adaptation doubled top-3 recall on untouched future events using only a small set of early records.")
        ],
        ref_des="U1 [PROVEN]"
    )

    b.add_card_text_block(
        slide, Inches(6.86), Inches(1.75), Inches(5.66), Inches(4.9),
        title="PRACTICAL BOUNDARIES",
        items=[
            ("Not Fully Unattended", "We do not recommend fully automated dispatch without technician oversight. Edge cases should always route to human engineers."),
            ("Fluid Power Excluded", "Our tests focused on rotating machinery, pumps, and compressors. Hydraulic fluid power loops require dedicated calibration."),
            ("Plant-Specific Base Rates", "While zero-shot transfer works, calibrating base rates with early plant records is strongly advised for best performance.")
        ],
        ref_des="U2 [BOUNDS]"
    )

    b.add_copper_trace_45(slide, Inches(6.46), Inches(4.0), Inches(6.86), Inches(4.0), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "Slide 21 lays out what the system can and cannot do. We proved that cross-asset transfer works, early warnings catch "
        "breakdowns, and early records adapt the model quickly. At the same time, we do not claim the system should run without "
        "human oversight. It is designed to assist dispatchers and technicians, with ambiguous cases routed to senior engineers."
    )


def build_slide_22(b, slide):
    """Slide 22: Summary and Journal Submission Roadmap with Radar Figure."""
    b.add_circuit_header(
        slide,
        eyebrow="SUMMARY // ROADMAP",
        title="Project contributions and roadmap for journal submission",
        subtitle="FINAL ARTIFACTS COMPLETED AND READY FOR IEEE TRANSACTIONS SUBMISSION",
        bus_id="BUS_ROADMAP"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(4.8), Inches(4.9),
        title="PROJECT CONTRIBUTIONS",
        stat_callout=("26 SECTIONS", "16 FIGURES"),
        items=[
            ("Complete Ingestion-to-Action Pipeline", "Built an automated workflow converting heterogeneous sensor streams into prioritized, feasible repair orders."),
            ("Realistic Plant Validation", "Benchmarked models on real factory slurry pumps rather than relying solely on synthetic failure generators."),
            ("Fast Adaptation Method", "Created a lightweight action prior update that doubles top-3 repair coverage without retraining heavy decision trees."),
            ("Target Submission", "Targeting submission to IEEE Transactions on Industrial Informatics (IEEE TII).")
        ],
        ref_des="U1 [CONTRIBUTION-SPEC]"
    )

    b.add_circuit_figure_card(
        slide, Inches(5.9), Inches(1.75), Inches(6.633), Inches(4.9),
        image_path="outputs/phase5/final_refinement/figures/fig16_comprehensive_radar_summary.png",
        title="PRESCRIPTIVE SYSTEM CAPABILITY RADAR",
        ref_des="DISP_01 [RADAR-PLOT]",
        caption="Multi-dimensional capability comparison between frozen zero-shot and adapted hybrid configurations"
    )

    b.add_copper_trace_45(slide, Inches(5.6), Inches(4.2), Inches(5.9), Inches(4.2), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "To wrap up, the project delivers a complete prescriptive maintenance system that turns noisy telemetry into concrete "
        "repair actions. On the right, our multi-dimensional radar plot summarizes the system across category agreement, "
        "MRR, component identification, failure anticipation, selective coverage, and cost savings. All research artifacts are "
        "complete, and we are preparing the manuscript for submission to IEEE Transactions on Industrial Informatics."
    )


def build_slide_23(b, slide):
    """Slide 23: Committee Q&A."""
    b.add_circuit_header(
        slide,
        eyebrow="DISCUSSION // COMMITTEE QUESTIONS",
        title="Key technical discussion points and common questions",
        subtitle="PRACTICAL ANSWERS ON MODEL CHOICES, DATA QUALITY, AND RELIABILITY",
        bus_id="BUS_QA"
    )

    b.add_card_text_block(
        slide, Inches(0.8), Inches(1.75), Inches(3.64), Inches(4.9),
        title="Q1: PRIOR TUNING",
        items=[
            ("Question", "Why adjust action priors instead of fine-tuning the decision trees on factory work orders?"),
            ("Answer", "Retraining gradient trees on just 63 work orders leads to severe overfitting. Updating action priors takes seconds, keeps the ranking model stable, and doubled top-3 recall on future events.")
        ],
        ref_des="U1 [QA-ADAPT]"
    )

    b.add_card_text_block(
        slide, Inches(4.84), Inches(1.75), Inches(3.64), Inches(4.9),
        title="Q2: DATASET CHOICE",
        items=[
            ("Question", "Why disqualify the Microsoft Azure Predictive Maintenance dataset?"),
            ("Answer", "Azure PM relies on generated synthetic failure rules and arbitrary simulated error flags. It has no real mechanic interventions. Testing against it would produce misleading numbers.")
        ],
        ref_des="U2 [QA-AZURE]"
    )

    b.add_card_text_block(
        slide, Inches(8.88), Inches(1.75), Inches(3.64), Inches(4.9),
        title="Q3: DATA INTEGRITY",
        items=[
            ("Question", "How can you be sure no future failure data leaked into recommendations?"),
            ("Answer", "Every sensor window ended strictly before the recommendation time. Model weights were locked before testing, and 14 automated audit checks verified zero lookahead leakage.")
        ],
        ref_des="U3 [QA-LEAKAGE]"
    )

    b.add_copper_trace_45(slide, Inches(4.44), Inches(3.8), Inches(4.84), Inches(3.8), via_start=True, via_end=True)
    b.add_copper_trace_45(slide, Inches(8.48), Inches(3.8), Inches(8.88), Inches(3.8), via_start=True, via_end=True)

    b.add_speaker_notes(
        slide,
        "Slide 23 addresses three practical questions you might have: why we adjusted action priors instead of retraining the trees, "
        "why we skipped the synthetic Azure dataset, and how we guaranteed that future failure data did not leak into the inputs. "
        "Thank you, Professor, for your time and feedback throughout this project. I am happy to answer any questions."
    )
