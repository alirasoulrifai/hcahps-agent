from io import BytesIO
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from textwrap import wrap
from reportlab.lib.utils import ImageReader
from reportlab.lib.utils import simpleSplit
import streamlit as st

@st.cache_data
def get_agent_df(df):
    """
    Prepares a user-friendly DataFrame for the Code Interpreter.
    Pivots the long-format HCAHPS data into a wide format with clear column names.
    """
    mask = df['measure_id'].str.contains("STAR_RATING", na=False)
    sub = df[mask].copy()
    
    # Ensure star_rating is numeric
    sub['star_rating'] = pd.to_numeric(sub['star_rating'], errors='coerce')

    # Pivot: One row per hospital
    pivoted = sub.pivot_table(
        index=['Facility ID', 'Facility Name', 'State', 'City/Town'], 
        columns='measure_id', 
        values='star_rating',
        aggfunc='first'
    ).reset_index()
    
    # Rename columns for friendliness
    rename_map = {
        'H_CLEAN_STAR_RATING': 'Cleanliness',
        'H_QUIET_STAR_RATING': 'Quietness',
        'H_COMP_1_STAR_RATING': 'Nurse Communication',
        'H_COMP_2_STAR_RATING': 'Doctor Communication',
        'H_COMP_3_STAR_RATING': 'Staff Responsiveness',
        'H_HSP_RATING_STAR_RATING': 'Overall Rating',
        'H_RECMND_STAR_RATING': 'Recommendation',
        'H_COMP_5_STAR_RATING': 'Medicine Communication',
        'H_COMP_6_STAR_RATING': 'Discharge Info',
        'H_COMP_7_STAR_RATING': 'Care Transition'
    }
    pivoted.rename(columns=rename_map, inplace=True)
    return pivoted


def create_pdf_report(question: str,
                      answer: str,
                      bar_img_bytes: BytesIO | None = None,
                      radar_img_bytes: BytesIO | None = None,
                      regulation_text: str | None = None,
                      ) -> BytesIO:
    """
    Build a PDF with the question, answer,
    and optional bar + radar chart images.
    Returns a BytesIO buffer ready for download.
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Margins
    margin_x = 40
    margin_y = 40
    text_width = width - 2 * margin_x
    y = height - margin_y

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_x, y, "HCAHPS RAG + Agent Report")
    y -= 30

    # Question
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "Question:")
    y -= 18
    c.setFont("Helvetica", 11)
    for line in wrap(question, 90):
        c.drawString(margin_x, y, line)
        y -= 14

    y -= 10

    # Answer
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "Answer:")
    y -= 18
    c.setFont("Helvetica", 11)
    
    # Handle long answers with pagination
    wrapper = wrap(answer, 95)
    for line in wrapper:
        if y < 60:
            c.showPage()
            y = height - margin_y
            c.setFont("Helvetica", 11)
        c.drawString(margin_x, y, line)
        y -= 14

    # ----- REGULATION SECTION -----
    if regulation_text:
        y -= 20
        if y < 100:
             c.showPage()
             y = height - margin_y
             
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin_x, y, "Relevant Federal Regulations")
        y -= 20

        c.setFont("Helvetica", 10)
        max_width = 500  # adjust based on page size

        for line in regulation_text.split("\n"):
            wrapped_lines = simpleSplit(line, "Helvetica", 10, max_width)
            for wline in wrapped_lines:
                if y < 50:
                    c.showPage()
                    y = height - margin_y
                    c.setFont("Helvetica", 10)
                c.drawString(margin_x, y, wline)
                y -= 14
            y -= 6

    # --- BAR CHART PAGE ---
    if bar_img_bytes is not None:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin_x, height - margin_y, "Bar chart for retrieved facilities")
        
        try:
            bar_img = ImageReader(bar_img_bytes)
            c.drawImage(
                bar_img,
                margin_x,
                150,
                width=width - 2 * margin_x,
                preserveAspectRatio=True,
                mask="auto",
            )
        except Exception:
            pass # Invalid image data handling

    # --- RADAR CHART PAGE ---
    if radar_img_bytes is not None:
        if bar_img_bytes is None: # Only show new page if we haven't already for bar chart, mostly likely separate pages is better actually
             c.showPage()
        else:
             c.showPage() # Always new page for radar for clarity
             
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin_x, height - margin_y, "Radar chart for selected facility")

        try:
            radar_img = ImageReader(radar_img_bytes)
            c.drawImage(
                radar_img,
                margin_x,
                150,
                width=width - 2 * margin_x,
                preserveAspectRatio=True,
                mask="auto",
            )
        except Exception:
            pass

    c.save()
    buf.seek(0)
    return buf
