from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "portfolio_credit_risk_analytics_ml_scoring.pdf"

NAVY = colors.HexColor("#172052")
PURPLE = colors.HexColor("#6D45F6")
BLUE = colors.HexColor("#2F68F6")
GREEN = colors.HexColor("#29B878")
AMBER = colors.HexColor("#F6B537")
RED = colors.HexColor("#FF4D64")
TEXT = colors.HexColor("#182235")
MUTED = colors.HexColor("#5D687A")
LIGHT = colors.HexColor("#F4F6FA")
LINE = colors.HexColor("#D8DEE9")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            textColor=NAVY,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=MUTED,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=NAVY,
            spaceBefore=6,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=13.2,
            textColor=TEXT,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=11,
            textColor=MUTED,
        ),
        "metric_num": ParagraphStyle(
            "metric_num",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=21,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "metric_lbl": ParagraphStyle(
            "metric_lbl",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "white": ParagraphStyle(
            "white",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "center": ParagraphStyle(
            "center",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=TEXT,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=NAVY,
        ),
    }


S = styles()


def p(text, style="body"):
    return Paragraph(text, S[style])


def bullet(text):
    return Paragraph(f"&bull; {text}", S["body"])


def metric_card(number, label, color=NAVY):
    data = [[p(number, "metric_num")], [p(label, "metric_lbl")]]
    t = Table(data, colWidths=[1.38 * inch], rowHeights=[0.34 * inch, 0.42 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.75, LINE),
                ("LINEABOVE", (0, 0), (-1, 0), 4, color),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def section_band(text):
    t = Table([[p(text, "white")]], colWidths=[7.28 * inch], rowHeights=[0.38 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return t


def simple_table(rows, widths, header=True):
    data = [[p(str(cell), "body") for cell in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
            ("LINEABOVE", (0, 0), (-1, 0), 2, NAVY),
        ]
        for c in range(len(rows[0])):
            data[0][c] = Paragraph(f"<b>{rows[0][c]}</b>", S["table_header"])
    t.setStyle(TableStyle(style))
    return t


def fit_image(relative_path, max_w, max_h):
    path = ROOT / relative_path
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 0.47 * inch, A4[0] - doc.rightMargin, 0.47 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.32 * inch, "Credit Risk Analytics & ML Scoring Pipeline | Portfolio Work Sample")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.32 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build():
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=0.48 * inch,
        leftMargin=0.48 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.62 * inch,
        title="Credit Risk Analytics & ML Scoring Pipeline - Portfolio",
        author="Nguyen Pham Khoi Nguyen",
    )

    story = []
    repo_url = "https://github.com/susayold/credit-risk-analytics-ml-scoring"

    story += [
        p("Credit Risk Analytics & ML Scoring Pipeline", "title"),
        p(
            "Portfolio work sample | Python, SQL-style ETL, Power BI, LightGBM, Logistic Regression, SHAP | May-Jun 2026",
            "subtitle",
        ),
        p(
            f"<b>Repository:</b> <link href='{repo_url}' color='blue'>{repo_url}</link> "
            "(private repository; access available on request).",
            "body",
        ),
        Spacer(1, 8),
    ]

    metrics = [
        metric_card("307K+", "labeled loan applications", BLUE),
        metric_card("271", "customer-level features", PURPLE),
        metric_card("25.5%", "default rate: CC util >100%", RED),
        metric_card("3.66x", "ML Lift@10", GREEN),
        metric_card("67.9%", "defaults captured in top 29.6%", AMBER),
    ]
    story.append(Table([metrics], colWidths=[1.42 * inch] * 5))
    story += [Spacer(1, 14), section_band("Project Summary"), Spacer(1, 8)]
    story += [
        p(
            "Built an end-to-end credit risk analytics pipeline on Home Credit loan application data. "
            "The project transforms raw application and historical credit tables into a customer-level master table, "
            "then uses the result for descriptive analytics, Power BI dashboarding, diagnostic modeling, ML scoring, "
            "risk banding, and model governance.",
            "body",
        ),
        p("<b>Business objective:</b> reduce wrong approval decisions by prioritizing high-risk applications for review while keeping human-in-the-loop controls.", "body"),
    ]

    deliverables = [
        ["Deliverable", "What it demonstrates"],
        ["SQL ETL layer", "Cleaning flags, missing-value flags, feature engineering, joins, and customer-level aggregation by SK_ID_CURR."],
        ["Python analytics pipeline", "Data preparation, descriptive statistics, correlation, diagnostic analytics, model benchmarking, and SHAP explainability."],
        ["Power BI dashboard", "Default-rate monitoring across borrower profile, credit burden, bureau history, payment behavior, and credit card usage."],
        ["ML risk scoring", "LightGBM champion model and Logistic Regression diagnostic model for business-readable risk drivers."],
    ]
    story += [Spacer(1, 6), simple_table(deliverables, [1.55 * inch, 5.55 * inch]), PageBreak()]

    story += [section_band("Technical Pipeline"), Spacer(1, 10)]
    flow = [
        ["Stage", "Main work", "Evidence in repository"],
        ["Data understanding", "Profile application and historical credit tables; define grain and target.", "src/step02_data_understanding.py"],
        ["Cleaning & feature flags", "Create missing flags, special-value handling, safe ratios, and application-level features.", "src/step03_cleaning_feature_flags.py; sql/02-03"],
        ["Descriptive analytics", "Full statistics, correlations, quantile/bin analysis, and dashboard source tables.", "src/step04_*; outputs/tables/step04_*"],
        ["Master aggregation", "Aggregate bureau, previous application, POS, installment, and credit card history to customer level.", "src/step05_build_master_table.py; sql/04-06"],
        ["Dashboard & diagnostics", "Power BI insight layer plus 28-variable Logistic Regression diagnostic model.", "dashboard/*.pbix; outputs/tables/step07_*"],
        ["ML & governance", "LightGBM scoring, PR-AUC/ROC-AUC/Lift, risk bands, SHAP, fairness/proxy sensitivity checks.", "notebooks/step08_machine_learning_governance.ipynb"],
    ]
    story += [simple_table(flow, [1.35 * inch, 3.05 * inch, 2.7 * inch]), Spacer(1, 12)]
    story += [p("<b>Master table:</b> final processed output contains 271 customer-level features and supports dashboarding, diagnostic analysis, and ML scoring.", "body")]
    story += [p("<b>SQL scope:</b> SQL covers the data engineering layer; Python remains the modeling and explainability layer.", "body")]
    story += [PageBreak()]

    story += [section_band("Business Insight Evidence"), Spacer(1, 10)]
    story += [
        p(
            "A key finding is that credit card over-utilization is a clear risk signal. "
            "Customers with utilization above 100% show a 25.5% observed default rate, compared with the 8.07% portfolio baseline.",
            "body",
        ),
        fit_image("outputs/figures/step06_dashboard/11_default_rate_by_credit_card_utilization.png", 6.7 * inch, 3.05 * inch),
        Spacer(1, 8),
    ]
    insight_table = [
        ["Finding", "Business interpretation"],
        ["Credit card utilization >100%: 25.5% default rate", "Strong stress signal; customers are using beyond limit and should be reviewed more tightly."],
        ["Risk is 3.16x above the 8.07% baseline", "The segment is much riskier than the portfolio average, so it is useful for policy and dashboard monitoring."],
        ["External scores and payment behavior recur in analysis", "The ML layer agrees with dashboard and diagnostic insights: external risk scores, debt burden, and repayment behavior matter."],
    ]
    story += [simple_table(insight_table, [2.35 * inch, 4.75 * inch]), PageBreak()]

    story += [section_band("Modeling Results & Review Prioritization"), Spacer(1, 10)]
    story += [
        p(
            "The ML layer is designed for prioritization, not automatic rejection. "
            "The model ranks applications by risk so credit teams can focus review capacity where the observed default concentration is highest.",
            "body",
        )
    ]
    model_metrics = [
        metric_card("0.7907", "Validation ROC-AUC", BLUE),
        metric_card("0.3127", "Validation PR-AUC", RED),
        metric_card("3.66x", "Lift@10", GREEN),
        metric_card("29.6%", "highest-risk applications reviewed", PURPLE),
        metric_card("2.27x", "review efficiency vs random", AMBER),
    ]
    story.append(Table([model_metrics], colWidths=[1.42 * inch] * 5))
    story += [Spacer(1, 12)]
    story += [
        fit_image("outputs/figures/step08_ml/08_step8_figures/v3_advanced_cell14_img01.png", 6.8 * inch, 2.55 * inch),
        Spacer(1, 8),
        fit_image("outputs/figures/step08_ml/08_step8_figures/raw_benchmark_cell11_img01.png", 4.95 * inch, 2.0 * inch),
        Spacer(1, 8),
        p(
            "<b>Decision framing:</b> green band = fast approval, amber band = manual review, red band = strict review/control. "
            "Reviewing the top 29.6% highest-risk applications captures about 67.9% of default cases.",
            "body",
        ),
        PageBreak(),
    ]

    story += [section_band("Explainability, Governance, and Portfolio Fit"), Spacer(1, 10)]
    story += [
        p(
            "Model explainability was handled with SHAP to show which features contributed most to model output. "
            "The strongest signals included external score summaries, annuity/credit burden, organization type, and historical credit behavior.",
            "body",
        ),
        fit_image("outputs/figures/step08_ml/08_step8_figures/v3_advanced_cell14_img05.png", 4.7 * inch, 5.0 * inch),
        Spacer(1, 8),
    ]
    gov = [
        ["Governance area", "How it was handled"],
        ["Sensitive/proxy features", "Occupation, organization, and gender-related signals were tested through sensitivity and fairness-gap checks."],
        ["Human-in-the-loop", "Risk bands support review prioritization; the score should not be used as a fully automatic rejection rule."],
        ["Business readiness", "The outputs connect analytics, dashboard, diagnostic modeling, ML scoring, and governance into one workflow."],
    ]
    story += [simple_table(gov, [1.8 * inch, 5.3 * inch]), Spacer(1, 10)]
    story += [
        KeepTogether(
            [
                p("<b>What this project shows I can do:</b>", "h2"),
                bullet("Build a reproducible analytics pipeline from multi-table raw data to a customer-level analytical dataset."),
                bullet("Translate credit-risk data into dashboards, segments, model metrics, and business review actions."),
                bullet("Use SQL for ETL logic, Python for analytics/modeling, and Power BI for stakeholder-facing insight."),
                bullet("Communicate model performance using ROC-AUC, PR-AUC, Lift@10, risk bands, SHAP, and governance checks."),
            ]
        )
    ]

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUT)


if __name__ == "__main__":
    build()
