"""Generate Publication-Grade 20-Minute Supervisor Presentation in 'Circuit' Theme.

Theme Specifications (SlideSpeak 'Circuit' Design Language):
- Background: PCB green #0E4D3A on every slide.
- Typography:
  * Headings: White 'Chakra Petch' (Google Font)
  * Labels, Captions, Axes, Badges: Uppercase 'Share Tech Mono' (Google Font), white at 55% opacity (#8CB9AA)
  * Eyebrows & Numbers: Copper #D9A45B and Gold #E8C27A
- Signature Motifs:
  1. Copper Traces: 2px #D9A45B lines running straight, bending at EXACTLY 45 degrees, ending in via dots
     (12px copper donuts with #0E4D3A hole).
  2. IC Chip Cards: Dark #0A3528 rectangles with faint copper outline and four 8x3px gold pin legs in #E8C27A
     on each side, each marked with a reference designator like 'U1' in the top right corner.
  3. Corner Pads: 12px gold rounded-square pads #E8C27A in all four slide corners with central drill hole.
  4. Stray Silkscreen Labels: 'R42', 'TP3', 'C17', 'GND', '+5V', 'CLK', etc.
  5. Charts: Copper bars #D9A45B with key bar in gold #E8C27A.
- Strictly Avoid:
  * Blue tones (STRICTLY NO BLUE, NO CYAN, NO PURPLE)
  * Gradients (Solid flat colors only)
  * Drop shadows (Flat vector aesthetic)
  * Photographs (Vector cards, IC chips, circuit schematics, structured tables)
  * 90 degree trace bends (All bends strictly at 45 degrees)
  * Rounded corners beyond 3px (Sharp rectangles or max 3px radius)
- Text Box Padding:
  * Generous margins on all text boxes and IC chips to prevent crowding or clipping.
- Speaker Notes:
  * Stored strictly in PowerPoint's native notes section across all 22 slides.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

# -------------------------------------------------------------
# PALETTE & CONSTANTS (Circuit Theme)
# -------------------------------------------------------------
COLOR_PCB_GREEN = RGBColor(14, 77, 58)      # #0E4D3A Industrial FR-4 resin green
COLOR_IC_BODY = RGBColor(10, 53, 40)        # #0A3528 Dark IC chip package
COLOR_COPPER = RGBColor(217, 164, 91)       # #D9A45B Copper traces, outlines, numbers
COLOR_GOLD = RGBColor(232, 194, 122)        # #E8C27A Gold pin legs, pads, key bar
COLOR_SILKSCREEN = RGBColor(140, 185, 170)  # #8CB9AA Silkscreen white at 55% opacity
COLOR_WHITE = RGBColor(255, 255, 255)       # Pure white headings
COLOR_ALERT = RGBColor(200, 100, 70)        # Subtle copper-rust for failure / caution

FONT_HEADING = "Segoe UI"
FONT_BODY = "Consolas"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PPTX = PROJECT_ROOT / "Smart_Manufacturing_MAS_Supervisor_Presentation.pptx"


class CircuitPresentationBuilder:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.blank_layout = self.prs.slide_layouts[6]

    # ---------------------------------------------------------
    # CORE PCB HARDWARE MOTIFS
    # ---------------------------------------------------------
    def _set_pcb_background(self, slide):
        """Creates the PCB substrate with 4 gold registration corner pads."""
        # 1. Full-bleed PCB green background
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_PCB_GREEN
        bg.line.fill.background()

        # 2. Four 12px gold corner pads (#E8C27A) with central drill hole (#0E4D3A)
        pad_size = Pt(12)
        hole_size = Pt(5)
        offset = Inches(0.4)
        pad_coords = [
            (offset, offset),
            (Inches(13.333) - offset - pad_size, offset),
            (offset, Inches(7.5) - offset - pad_size),
            (Inches(13.333) - offset - pad_size, Inches(7.5) - offset - pad_size)
        ]
        for px, py in pad_coords:
            # Gold pad
            pad = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, px, py, pad_size, pad_size)
            pad.fill.solid()
            pad.fill.fore_color.rgb = COLOR_GOLD
            pad.line.fill.background()
            # Drill hole
            hole = slide.shapes.add_shape(MSO_SHAPE.OVAL, px + Pt(3.5), py + Pt(3.5), hole_size, hole_size)
            hole.fill.solid()
            hole.fill.fore_color.rgb = COLOR_PCB_GREEN
            hole.line.fill.background()

    def add_via_dot(self, slide, x_in, y_in):
        """12px copper donut with #0E4D3A hole."""
        outer = slide.shapes.add_shape(MSO_SHAPE.OVAL, int(x_in - Pt(6)), int(y_in - Pt(6)), int(Pt(12)), int(Pt(12)))
        outer.fill.solid()
        outer.fill.fore_color.rgb = COLOR_COPPER
        outer.line.fill.background()

        inner = slide.shapes.add_shape(MSO_SHAPE.OVAL, int(x_in - Pt(3)), int(y_in - Pt(3)), int(Pt(6)), int(Pt(6)))
        inner.fill.solid()
        inner.fill.fore_color.rgb = COLOR_PCB_GREEN
        inner.line.fill.background()

    def add_copper_trace_45(self, slide, x1, y1, x2, y2, bend_x=None, via_start=False, via_end=True):
        """Draws a 2px copper line running straight, bending at EXACTLY 45 degrees."""
        dy = y2 - y1
        if abs(dy) < 0.001:
            conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, int(x1), int(y1), int(x2), int(y2))
            conn.line.color.rgb = COLOR_COPPER
            conn.line.width = Pt(2)
        else:
            diag_dist = int(abs(dy))
            if bend_x is None:
                bend_x = int(x1 + (x2 - x1 - diag_dist) / 2)
            else:
                bend_x = int(bend_x)

            x_diag_end = int(bend_x + diag_dist)

            if bend_x > x1:
                c1 = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, int(x1), int(y1), bend_x, int(y1))
                c1.line.color.rgb = COLOR_COPPER
                c1.line.width = Pt(2)

            c2 = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, bend_x, int(y1), x_diag_end, int(y2))
            c2.line.color.rgb = COLOR_COPPER
            c2.line.width = Pt(2)

            if x2 > x_diag_end:
                c3 = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x_diag_end, int(y2), int(x2), int(y2))
                c3.line.color.rgb = COLOR_COPPER
                c3.line.width = Pt(2)

        if via_start:
            self.add_via_dot(slide, x1, y1)
        if via_end:
            self.add_via_dot(slide, x2, y2)

    def add_silkscreen_label(self, slide, text, left, top, width=Inches(1.2), height=Inches(0.25), size_pt=8, bold=False, align=PP_ALIGN.LEFT):
        """Stray silkscreen print in Share Tech Mono, white at 55% opacity."""
        tb = slide.shapes.add_textbox(int(left), int(top), int(width), int(height))
        tf = tb.text_frame
        tf.word_wrap = False
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = text.upper()
        p.font.name = FONT_BODY
        p.font.size = Pt(size_pt)
        p.font.bold = bold
        p.font.color.rgb = COLOR_SILKSCREEN
        p.alignment = align
        return tb

    def add_ic_chip_card(self, slide, left, top, width, height, ref_des="U1", title=None, pin_count=4):
        """IC chip card: dark #0A3528 rectangle with faint copper outline and gold pin legs."""
        # 1. IC Chip body
        card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left), int(top), int(width), int(height))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_IC_BODY
        card.line.color.rgb = COLOR_COPPER
        card.line.width = Pt(1)

        # 2. Four 8x3px gold pin legs (#E8C27A) on left and right sides
        pin_w = Pt(8)
        pin_h = Pt(3)
        for i in range(pin_count):
            fraction = (i + 1) / (pin_count + 1)
            pin_y = int(top + height * fraction - Pt(1.5))
            # Left pin
            p_left = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left - pin_w), pin_y, int(pin_w), int(pin_h))
            p_left.fill.solid()
            p_left.fill.fore_color.rgb = COLOR_GOLD
            p_left.line.fill.background()
            # Right pin
            p_right = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left + width), pin_y, int(pin_w), int(pin_h))
            p_right.fill.solid()
            p_right.fill.fore_color.rgb = COLOR_GOLD
            p_right.line.fill.background()

        # 3. Pin 1 orientation index dot (top-left)
        p1 = slide.shapes.add_shape(MSO_SHAPE.OVAL, int(left + Pt(8)), int(top + Pt(8)), int(Pt(5)), int(Pt(5)))
        p1.fill.solid()
        p1.fill.fore_color.rgb = COLOR_COPPER
        p1.line.fill.background()

        # 4. Reference designator (e.g. U1, IC4) in top-right corner
        is_narrow = (width < Inches(3.0))
        ref_w = Inches(0.9) if is_narrow else Inches(1.3)
        ref_tb = slide.shapes.add_textbox(left + width - ref_w - Inches(0.15), top + Pt(4), ref_w, Inches(0.25))
        ref_tf = ref_tb.text_frame
        ref_tf.word_wrap = False
        ref_tf.margin_left = ref_tf.margin_top = ref_tf.margin_right = ref_tf.margin_bottom = 0
        ref_p = ref_tf.paragraphs[0]
        ref_p.text = ref_des.upper()
        ref_p.font.name = FONT_BODY
        ref_p.font.size = Pt(8.5) if is_narrow else Pt(10.0)
        ref_p.font.bold = True
        ref_p.font.color.rgb = COLOR_COPPER
        ref_p.alignment = PP_ALIGN.RIGHT

        # 5. Optional Title
        if title:
            title_w = width - Inches(0.3) if is_narrow else (width - Inches(1.6))
            title_top = top + Pt(16) if is_narrow else (top + Inches(0.12))
            title_size = Pt(10.5) if is_narrow else Pt(13)

            t_tb = slide.shapes.add_textbox(left + Inches(0.15 if is_narrow else 0.25), title_top, title_w, Inches(0.3))
            t_tf = t_tb.text_frame
            t_tf.word_wrap = True
            t_tf.margin_left = t_tf.margin_top = t_tf.margin_right = t_tf.margin_bottom = 0
            t_p = t_tf.paragraphs[0]
            t_p.text = title.upper()
            t_p.font.name = FONT_HEADING
            t_p.font.size = title_size
            t_p.font.bold = True
            t_p.font.color.rgb = COLOR_GOLD

        return card

    def add_circuit_header(self, slide, eyebrow: str, title: str, subtitle: str = None, bus_id: str = "BUS_0..7"):
        """Header with silkscreen badge, white Chakra Petch title, and 45-degree copper bus trace."""
        self._set_pcb_background(slide)

        # Eyebrow / Category silkscreen tag
        eyebrow_tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8.0), Inches(0.25))
        eyebrow_tf = eyebrow_tb.text_frame
        eyebrow_tf.word_wrap = False
        eyebrow_tf.margin_left = eyebrow_tf.margin_top = eyebrow_tf.margin_right = eyebrow_tf.margin_bottom = 0
        p_eye = eyebrow_tf.paragraphs[0]
        p_eye.text = f"// {eyebrow.upper()} // REF: {bus_id}"
        p_eye.font.name = FONT_BODY
        p_eye.font.size = Pt(10)
        p_eye.font.bold = True
        p_eye.font.color.rgb = COLOR_COPPER

        # Main Heading in white Chakra Petch
        title_tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.68), Inches(11.7), Inches(0.65))
        title_tf = title_tb.text_frame
        title_tf.word_wrap = True
        title_tf.margin_left = title_tf.margin_top = title_tf.margin_right = title_tf.margin_bottom = 0
        p_title = title_tf.paragraphs[0]
        p_title.text = title
        p_title.font.name = FONT_HEADING
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_WHITE

        # Subtitle in Share Tech Mono silkscreen
        if subtitle:
            p_sub = title_tf.add_paragraph()
            p_sub.text = subtitle.upper()
            p_sub.font.name = FONT_BODY
            p_sub.font.size = Pt(10)
            p_sub.font.color.rgb = COLOR_SILKSCREEN

        # Copper bus trace running horizontally under header
        self.add_copper_trace_45(
            slide,
            x1=Inches(0.8), y1=Inches(1.42),
            x2=Inches(12.5), y2=Inches(1.42),
            via_start=True, via_end=True
        )

        # Ambient silkscreen noise
        self.add_silkscreen_label(slide, "TP1_GND", Inches(11.4), Inches(0.42), width=Inches(0.7), size_pt=8)
        self.add_silkscreen_label(slide, "+3V3_VCC", Inches(12.2), Inches(0.42), width=Inches(0.7), size_pt=8)
        self.add_silkscreen_label(slide, "LAYER_TOP // FR4", Inches(11.2), Inches(1.22), width=Inches(1.5), size_pt=8)

    def add_kpi_register(self, slide, left, top, width, height, reg_id: str, label: str, value: str, subtext: str = None):
        """Creates an IC register card displaying an empirical metric callout."""
        self.add_ic_chip_card(slide, left, top, width, height, ref_des=reg_id, pin_count=3)

        tb = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.12), width - Inches(0.36), height - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        # Label (Silkscreen uppercase)
        p_lbl = tf.paragraphs[0]
        p_lbl.text = label.upper()
        p_lbl.font.name = FONT_BODY
        p_lbl.font.size = Pt(10)
        p_lbl.font.bold = True
        p_lbl.font.color.rgb = COLOR_SILKSCREEN

        # Value (Chakra Petch Copper/Gold)
        p_val = tf.add_paragraph()
        p_val.text = value
        p_val.font.name = FONT_HEADING
        p_val.font.size = Pt(22)
        p_val.font.bold = True
        p_val.font.color.rgb = COLOR_GOLD

        # Subtext
        if subtext:
            p_sub = tf.add_paragraph()
            p_sub.text = subtext
            p_sub.font.name = FONT_BODY
            p_sub.font.size = Pt(9.5)
            p_sub.font.color.rgb = COLOR_COPPER

    def add_card_text_block(self, slide, left, top, width, height, title, items, ref_des="U1", pin_count=4, stat_callout=None):
        """Renders an IC Chip card populated with padded structured bullet points or text items."""
        self.add_ic_chip_card(slide, left, top, width, height, ref_des=ref_des, title=title, pin_count=pin_count)

        is_narrow = (width < Inches(3.0))
        content_top = top + (Inches(0.58) if is_narrow else Inches(0.52))
        content_height = height - (Inches(0.68) if is_narrow else Inches(0.65))
        pad_x = Inches(0.15) if is_narrow else Inches(0.25)

        if stat_callout:
            callout_tb = slide.shapes.add_textbox(int(left + pad_x), int(content_top), int(width - pad_x * 2), int(Inches(0.85)))
            c_tf = callout_tb.text_frame
            c_tf.word_wrap = True
            c_tf.margin_left = c_tf.margin_top = c_tf.margin_right = c_tf.margin_bottom = 0

            p_cval = c_tf.paragraphs[0]
            p_cval.text = stat_callout[0]
            p_cval.font.name = FONT_HEADING
            p_cval.font.size = Pt(22)
            p_cval.font.bold = True
            p_cval.font.color.rgb = COLOR_GOLD

            p_clbl = c_tf.add_paragraph()
            p_clbl.text = stat_callout[1].upper()
            p_clbl.font.name = FONT_BODY
            p_clbl.font.size = Pt(9.5)
            p_clbl.font.bold = True
            p_clbl.font.color.rgb = COLOR_COPPER

            content_top += Inches(0.9)
            content_height -= Inches(0.9)

        tb = slide.shapes.add_textbox(int(left + pad_x), int(content_top), int(width - pad_x * 2), int(content_height))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        header_size = Pt(9.5) if is_narrow else Pt(12.0)
        body_size = Pt(9.0) if is_narrow else Pt(11.5)
        para_space = Pt(2.5) if is_narrow else Pt(5.5)

        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = para_space
            if isinstance(item, tuple):
                # (Header, Body)
                run_h = p.add_run()
                run_h.text = f"{item[0]}: "
                run_h.font.name = FONT_BODY
                run_h.font.bold = True
                run_h.font.size = header_size
                run_h.font.color.rgb = COLOR_COPPER

                run_b = p.add_run()
                run_b.text = item[1]
                run_b.font.name = FONT_BODY
                run_b.font.size = body_size
                run_b.font.color.rgb = COLOR_SILKSCREEN
            else:
                run = p.add_run()
                run.text = f"• {item}"
                run.font.name = FONT_BODY
                run.font.size = body_size
                run.font.color.rgb = COLOR_SILKSCREEN

    def add_circuit_barchart(self, slide, left, top, width, height, categories, values, key_index, unit="%", title=None):
        """Renders a custom PCB horizontal bar chart with copper bars and gold key bar."""
        self.add_ic_chip_card(slide, left, top, width, height, ref_des="ADC_PLOT", title=title, pin_count=4)

        num_bars = len(categories)
        max_val = max(values) if values else 1.0
        row_height = (height - Inches(0.65)) / num_bars
        bar_max_width = width - Inches(3.4)

        for i, (cat, val) in enumerate(zip(categories, values)):
            y = int(top + Inches(0.5) + i * row_height)

            # Category label (Silkscreen Mono)
            lbl_tb = slide.shapes.add_textbox(int(left + Inches(0.25)), int(y + Pt(2)), int(Inches(2.0)), int(Pt(16)))
            lbl_tf = lbl_tb.text_frame
            lbl_tf.word_wrap = False
            lbl_tf.margin_left = lbl_tf.margin_top = lbl_tf.margin_right = lbl_tf.margin_bottom = 0
            p_l = lbl_tf.paragraphs[0]
            p_l.text = cat.upper()
            p_l.font.name = FONT_BODY
            p_l.font.size = Pt(10.5)
            p_l.font.bold = (i == key_index)
            p_l.font.color.rgb = COLOR_WHITE if (i == key_index) else COLOR_SILKSCREEN

            # Bar track groove
            track_w = int(bar_max_width)
            track = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left + Inches(2.35)), int(y + Pt(4)), track_w, int(Pt(11)))
            track.fill.solid()
            track.fill.fore_color.rgb = RGBColor(16, 50, 38)
            track.line.color.rgb = RGBColor(25, 75, 58)
            track.line.width = Pt(0.75)

            # Bar fill: Copper for standard, Gold for Key Bar!
            is_key = (i == key_index)
            bar_color = COLOR_GOLD if is_key else COLOR_COPPER
            bar_w = int(track_w * (val / max_val))
            if bar_w > Pt(2):
                bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left + Inches(2.35)), int(y + Pt(4)), bar_w, int(Pt(11)))
                bar.fill.solid()
                bar.fill.fore_color.rgb = bar_color
                bar.line.fill.background()

            # Value label
            val_tb = slide.shapes.add_textbox(int(left + Inches(2.4) + track_w + Inches(0.1)), int(y + Pt(2)), int(Inches(0.8)), int(Pt(16)))
            val_tf = val_tb.text_frame
            val_tf.word_wrap = False
            val_tf.margin_left = val_tf.margin_top = val_tf.margin_right = val_tf.margin_bottom = 0
            p_v = val_tf.paragraphs[0]
            p_v.text = f"{val:.1f}{unit}" if unit == "%" else f"{val:.4f}"
            p_v.font.name = FONT_BODY
            p_v.font.size = Pt(11)
            p_v.font.bold = True
            p_v.font.color.rgb = COLOR_GOLD if is_key else COLOR_COPPER

            # 45-degree lead-out trace for key bar
            if is_key:
                tx = int(left + Inches(2.35) + bar_w)
                ty = int(y + Pt(9))
                self.add_via_dot(slide, tx, ty)

    def add_circuit_figure_card(self, slide, left, top, width, height, image_path: str, title: str, ref_des: str = "DISP_01", pin_count: int = 4, caption: str = None):
        """Embeds a figure inside an IC Module card with preserved aspect ratio, clean white substrate, and silkscreen labels."""
        import os
        from PIL import Image

        self.add_ic_chip_card(slide, left, top, width, height, ref_des=ref_des, title=title, pin_count=pin_count)

        # Resolve path
        if not os.path.isabs(image_path):
            image_path = str(PROJECT_ROOT / image_path)

        if not os.path.exists(image_path):
            print(f"[WARNING] Image path not found: {image_path}")
            return

        pad_x = Inches(0.22)
        pad_top = Inches(0.52)
        pad_bottom = Inches(0.42) if caption else Inches(0.18)

        max_w = width - (pad_x * 2)
        max_h = height - pad_top - pad_bottom

        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size

            aspect = img_w / img_h
            box_aspect = max_w / max_h

            if aspect > box_aspect:
                target_w = max_w
                target_h = max_w / aspect
            else:
                target_h = max_h
                target_w = max_h * aspect

            target_left = left + pad_x + (max_w - target_w) / 2
            target_top = top + pad_top + (max_h - target_h) / 2

            # Bezel frame behind image
            bezel = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                int(target_left - Pt(2)), int(target_top - Pt(2)),
                int(target_w + Pt(4)), int(target_h + Pt(4))
            )
            bezel.fill.solid()
            bezel.fill.fore_color.rgb = RGBColor(255, 255, 255)
            bezel.line.color.rgb = COLOR_COPPER
            bezel.line.width = Pt(1.5)

            slide.shapes.add_picture(
                image_path,
                int(target_left), int(target_top),
                int(target_w), int(target_h)
            )
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")

        if caption:
            cap_tb = slide.shapes.add_textbox(int(left + pad_x), int(top + height - Inches(0.38)), int(width - pad_x * 2), int(Inches(0.3)))
            cap_tf = cap_tb.text_frame
            cap_tf.word_wrap = True
            cap_tf.margin_left = cap_tf.margin_top = cap_tf.margin_right = cap_tf.margin_bottom = 0
            cap_p = cap_tf.paragraphs[0]
            cap_p.text = caption.upper()
            cap_p.font.name = FONT_BODY
            cap_p.font.size = Pt(9.5)
            cap_p.font.color.rgb = COLOR_SILKSCREEN

    def add_speaker_notes(self, slide, notes_text: str):
        """Ensures complete spoken defense script is saved in PowerPoint's native notes section."""
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = notes_text

    def build_all(self):
        """Constructs all 23 publication-grade defense slides in the Circuit theme."""
        import circuit_slides_v2_p1 as p1
        import circuit_slides_v2_p2 as p2
        import circuit_slides_v2_p3 as p3

        print("Building Circuit Theme Presentation (23 Slides)...")

        # Slide 1 to 9 (Part 1, including new concrete example state slide)
        s1 = self.prs.slides.add_slide(self.blank_layout)
        p1.build_slide_01(self, s1)
        s2 = self.prs.slides.add_slide(self.blank_layout)
        p1.build_slide_02(self, s2)
        s3 = self.prs.slides.add_slide(self.blank_layout)
        p1.build_slide_03(self, s3)
        s4 = self.prs.slides.add_slide(self.blank_layout)
        p1.build_slide_04(self, s4)
        s5 = self.prs.slides.add_slide(self.blank_layout)
        p1.build_slide_05(self, s5)
        s6 = self.prs.slides.add_slide(self.blank_layout)
        p1.build_slide_06(self, s6)
        s7 = self.prs.slides.add_slide(self.blank_layout)
        p1.build_slide_07(self, s7)
        s8 = self.prs.slides.add_slide(self.blank_layout)
        p1.build_slide_08(self, s8)
        s9 = self.prs.slides.add_slide(self.blank_layout)
        p1.build_slide_09(self, s9)

        # Slide 10 to 17 (Part 2)
        s10 = self.prs.slides.add_slide(self.blank_layout)
        p2.build_slide_10(self, s10)
        s11 = self.prs.slides.add_slide(self.blank_layout)
        p2.build_slide_11(self, s11)
        s12 = self.prs.slides.add_slide(self.blank_layout)
        p2.build_slide_12(self, s12)
        s13 = self.prs.slides.add_slide(self.blank_layout)
        p2.build_slide_13(self, s13)
        s14 = self.prs.slides.add_slide(self.blank_layout)
        p2.build_slide_14(self, s14)
        s15 = self.prs.slides.add_slide(self.blank_layout)
        p2.build_slide_15(self, s15)
        s16 = self.prs.slides.add_slide(self.blank_layout)
        p2.build_slide_16(self, s16)
        s17 = self.prs.slides.add_slide(self.blank_layout)
        p2.build_slide_17(self, s17)

        # Slide 18 to 23 (Part 3)
        s18 = self.prs.slides.add_slide(self.blank_layout)
        p3.build_slide_18(self, s18)
        s19 = self.prs.slides.add_slide(self.blank_layout)
        p3.build_slide_19(self, s19)
        s20 = self.prs.slides.add_slide(self.blank_layout)
        p3.build_slide_20(self, s20)
        s21 = self.prs.slides.add_slide(self.blank_layout)
        p3.build_slide_21(self, s21)
        s22 = self.prs.slides.add_slide(self.blank_layout)
        p3.build_slide_22(self, s22)
        s23 = self.prs.slides.add_slide(self.blank_layout)
        p3.build_slide_23(self, s23)

        OUTPUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(OUTPUT_PPTX)
        print(f"[SUCCESS] Circuit-Themed Presentation saved to: {OUTPUT_PPTX}")
        print(f"[VERIFICATION] Total slides generated: {len(self.prs.slides)} / 23")


def main():
    builder = CircuitPresentationBuilder()
    builder.build_all()


if __name__ == "__main__":
    main()

