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
OUT = ROOT / "reports" / "portfolio_credit_risk_analytics_ml_scoring_detailed_v7.pdf"

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
            fontSize=23,
            leading=28,
            textColor=NAVY,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=14.5,
            textColor=MUTED,
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.3,
            leading=15.5,
            textColor=NAVY,
            spaceBefore=6,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.85,
            leading=12.4,
            textColor=TEXT,
            spaceAfter=5.5,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=10,
            textColor=MUTED,
        ),
        "metric_num": ParagraphStyle(
            "metric_num",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "metric_lbl": ParagraphStyle(
            "metric_lbl",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "white": ParagraphStyle(
            "white",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10.2,
            leading=13.5,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11.5,
            textColor=NAVY,
        ),
        "center": ParagraphStyle(
            "center",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12.5,
            textColor=TEXT,
            alignment=TA_CENTER,
        ),
    }


S = styles()


def p(text, style="body"):
    return Paragraph(text, S[style])


def bullet(text):
    return Paragraph(f"&bull; {text}", S["body"])


def metric_card(number, label, color=NAVY):
    data = [[p(number, "metric_num")], [p(label, "metric_lbl")]]
    t = Table(data, colWidths=[1.38 * inch], rowHeights=[0.34 * inch, 0.43 * inch])
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
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
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


def callout(text, color=NAVY):
    t = Table([[p(text, "body")]], colWidths=[7.1 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("LINEBEFORE", (0, 0), (0, -1), 4, color),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def conclusion(text, color=PURPLE):
    return callout(f"<b>Conclusion:</b> {text}", color)


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
    canvas.drawString(doc.leftMargin, 0.32 * inch, "Credit Risk Analytics & ML Scoring Pipeline | Detailed Portfolio Work Sample")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.32 * inch, f"Page {doc.page}")
    canvas.restoreState()


def add_front_page(story, repo_url):
    story += [
        p("Credit Risk Analytics & ML Scoring Pipeline", "title"),
        p(
            "Detailed portfolio work sample | Python, SQL-style ETL, Power BI, LightGBM, Logistic Regression, SHAP | May-Jun 2026",
            "subtitle",
        ),
        p(
            f"<b>Project repository:</b> <link href='{repo_url}' color='blue'>{repo_url}</link>",
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
    story += [Spacer(1, 14), section_band("1. Executive Summary"), Spacer(1, 8)]
    story += [
        p(
            "Built an end-to-end credit risk analytics pipeline on Home Credit loan application data. "
            "The project transforms raw application and historical credit tables into a customer-level master table, then uses the result for descriptive analytics, Power BI dashboarding, diagnostic modeling, ML scoring, risk banding, and model governance.",
            "body",
        ),
        p(
            "<b>Business objective:</b> reduce wrong approval decisions by prioritizing high-risk applications for review while keeping human-in-the-loop controls. The output is not designed to automatically reject customers; it turns model scores into review priority.",
            "body",
        ),
    ]
    deliverables = [
        ["My responsibility", "Concrete output"],
        ["Data engineering", "Created a customer-level analytical table with 271 features from application, bureau, previous application, POS, installment, and credit card history."],
        ["SQL ETL translation", "Documented SQL joins, cleaning flags, missing-value flags, feature engineering, historical aggregation, and master-table build logic."],
        ["Analytics and dashboarding", "Produced descriptive statistics, correlation/quantile analysis, risk segments, and Power BI-ready default-rate insights."],
        ["Modeling and governance", "Built diagnostic Logistic Regression and ML prioritization model; added SHAP, sensitivity, and fairness/proxy checks."],
    ]
    story += [
        Spacer(1, 6),
        simple_table(deliverables, [1.55 * inch, 5.55 * inch]),
        Spacer(1, 8),
        conclusion(
            "The project is designed as a decision-support workflow, not only a modeling exercise. "
            "Its value comes from connecting data engineering, dashboard insight, model validation, and operational risk bands into one consistent credit-risk process.",
            PURPLE,
        ),
        PageBreak(),
    ]


def add_dashboard_export_pages(story):
    dashboard_pages = [
        (
            "5.1 Power BI Dashboard - Overview",
            "Portfolio-level view: customer count, default rate, default customers, average credit amount, late payment share, target distribution, and a risk heatmap.",
            "reports/portfolio_assets/dashboard_page_01.png",
            "The overview page establishes the portfolio baseline and the key scale of the problem. "
            "It gives management a starting point: how many customers are being evaluated, what the overall default rate is, and which risk groups deserve deeper analysis.",
        ),
        (
            "5.2 Power BI Dashboard - Customer Profile",
            "Customer-profile view: age, education, income type, occupation, and demographic/profile segments connected to default rate.",
            "reports/portfolio_assets/dashboard_page_02.png",
            "Customer profile variables are useful for understanding portfolio composition, but they should not be used as stand-alone rejection rules. "
            "They are best interpreted as context that must be combined with repayment behavior, affordability, and historical credit signals.",
        ),
        (
            "5.3 Power BI Dashboard - Loan and Affordability",
            "Affordability view: credit amount, annuity, credit/income ratio, annuity/income ratio, contract type, and burden heatmap.",
            "reports/portfolio_assets/dashboard_page_03.png",
            "Affordability is a core credit-risk theme because it connects the loan obligation to repayment capacity. "
            "The page supports a practical review logic: raw loan amount is less informative than burden ratios such as credit/income and annuity/income.",
        ),
        (
            "5.4 Power BI Dashboard - Credit History",
            "Credit-history view: bureau loans, overdue share, previous refusal share, active loans, debt/credit pressure, and refusal/bureau risk heatmaps.",
            "reports/portfolio_assets/dashboard_page_04.png",
            "Historical credit behavior provides stronger evidence than static profile data. "
            "Previous refusals, overdue history, and bureau debt pressure help explain whether the customer has already shown stress in the credit system.",
        ),
        (
            "5.5 Power BI Dashboard - Payment Behavior",
            "Payment-behavior view: late payment, underpayment, credit card utilization, POS DPD, and repayment behavior risk heatmaps.",
            "reports/portfolio_assets/dashboard_page_05.png",
            "Payment behavior is one of the most decision-useful signals because it reflects what customers actually did, not only what they reported. "
            "Late payment, underpayment, and high card utilization should therefore be prioritized in manual review.",
        ),
        (
            "5.6 Power BI Dashboard - Risk Segmentation",
            "Risk-segmentation view: risk score groups, high-risk share, very-high-risk share, default rate by segment, and action recommendation summary.",
            "reports/portfolio_assets/dashboard_page_06.png",
            "The risk-segmentation page converts analysis into action. "
            "It connects model scores with business bands so that review resources can be focused on high-risk groups instead of being applied evenly across the portfolio.",
        ),
    ]
    for title, note, image_path, conclusion_text in dashboard_pages:
        story += [
            section_band(title),
            Spacer(1, 8),
            p(note, "body"),
            fit_image(image_path, 7.15 * inch, 4.35 * inch),
            Spacer(1, 8),
            conclusion(
                conclusion_text,
                BLUE,
            ),
            PageBreak(),
        ]


def build():
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=0.48 * inch,
        leftMargin=0.48 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.62 * inch,
        title="Credit Risk Analytics & ML Scoring Pipeline - Detailed Portfolio",
        author="Nguyen Pham Khoi Nguyen",
    )

    story = []
    repo_url = "https://github.com/susayold/credit-risk-analytics-ml-scoring"

    add_front_page(story, repo_url)

    story += [section_band("2. Business Problem and Decision Target"), Spacer(1, 10)]
    story += [
        p(
            "In credit approval, the business risk is two-sided: approving a truly high-risk customer increases credit loss, while rejecting or delaying a good customer hurts revenue and customer experience. The project therefore focuses on ranking risk and allocating review effort, not replacing credit policy.",
            "body",
        )
    ]
    problem = [
        ["Concept", "Definition used in the project"],
        ["TARGET = 1", "The customer had repayment difficulty under the dataset definition. It does not mean the customer certainly never pays; it is the modeled risk event."],
        ["Portfolio baseline", "Overall default rate used as the comparison point. The main baseline referenced in the project is 8.07%."],
        ["Risk score", "A model-generated ranking score. Higher score means higher estimated default risk relative to other applications."],
        ["Decision bands", "Green = process quickly, amber = manual review, red = stricter control. The model prioritizes review; final decision remains policy plus human judgment."],
    ]
    story += [simple_table(problem, [1.55 * inch, 5.55 * inch]), Spacer(1, 10)]
    story += [
        conclusion(
            "Credit risk should be framed as resource allocation, not only as default prediction. "
            "With an 8.07% portfolio baseline, random review would waste substantial effort on low-risk applications. "
            "The analytical objective is therefore to identify segments that are meaningfully above baseline and route them into the appropriate review path: faster processing, standard review, or stricter control.",
            PURPLE,
        ),
        PageBreak(),
    ]

    story += [section_band("3. Data Architecture and Feature Flow"), Spacer(1, 10)]
    flow = [
        ["Stage", "Main work", "Evidence in repository"],
        ["Data understanding", "Profile application and historical credit tables; define grain and target.", "src/step02_data_understanding.py"],
        ["Cleaning and feature flags", "Create missing flags, special-value handling, safe ratios, and application-level features.", "src/step03_cleaning_feature_flags.py; sql/02-03"],
        ["Descriptive analytics", "Full statistics, correlations, quantile/bin analysis, and dashboard source tables.", "src/step04_*; outputs/tables/step04_*"],
        ["Master aggregation", "Aggregate bureau, previous application, POS, installment, and credit card history to customer level.", "src/step05_build_master_table.py; sql/04-06"],
        ["Dashboard and diagnostics", "Power BI insight layer plus 28-variable Logistic Regression diagnostic model.", "dashboard/*.pbix; outputs/tables/step07_*"],
        ["ML and governance", "LightGBM scoring, PR-AUC/ROC-AUC/Lift, risk bands, SHAP, fairness/proxy sensitivity checks.", "notebooks/step08_machine_learning_governance.ipynb"],
    ]
    story += [simple_table(flow, [1.35 * inch, 3.05 * inch, 2.7 * inch]), Spacer(1, 12)]
    feature_flow = [
        ["Feature flow", "Count / logic"],
        ["Raw application_train.csv", "122 original application columns at loan-application grain."],
        ["Step 3 feature ownership", "+16 conceptual cleaning/application feature flags. In execution, these are materialized in the master-building script."],
        ["Step 5 master build", "+1 train/test alignment column, +126 historical aggregate features, +6 history coverage flags."],
        ["Final master table", "271 customer-level columns used as the source of truth for dashboard, diagnostic analysis, and ML scoring."],
    ]
    story += [simple_table(feature_flow, [2.05 * inch, 5.05 * inch]), Spacer(1, 8)]
    story += [
        p("<b>Important modeling scope:</b> the headline portfolio statistics are based on 307K+ labeled train applications. The unlabeled Kaggle test file is not used to compute default-rate performance.", "body"),
        conclusion(
            "The raw application table describes the borrower at application time, but credit risk is often embedded in historical behavior. "
            "Building a 271-feature customer-level master table brings application data, bureau history, previous applications, POS, installments, and credit card behavior into one consistent grain. "
            "This makes the downstream dashboard, diagnostic model, and ML model more reliable because they all read from the same business definitions and the same analytical base.",
            PURPLE,
        ),
        PageBreak(),
    ]

    story += [section_band("4. SQL ETL and Cleaning Logic"), Spacer(1, 10)]
    story += [
        p(
            "The original project was executed mainly with Python/pandas. I added a SQL ETL version to document how the data engineering layer can be implemented with SQL for technical handover and review.",
            "body",
        )
    ]
    sql_scope = [
        ["SQL file", "Purpose"],
        ["01_create_base_application.sql", "Create application-level base table and standard fields."],
        ["02_cleaning_missing_flags.sql", "Create missing flags, handle special values, and document cleaning assumptions."],
        ["03_feature_engineering_application.sql", "Build ratios and application-level engineered features such as credit/income and annuity/credit burden."],
        ["04_aggregate_bureau_and_bureau_balance.sql", "Aggregate external bureau history and monthly bureau balance to one row per customer."],
        ["05_aggregate_previous_pos_installments_credit_card.sql", "Aggregate previous applications, POS cash, installment payment, and credit card behavior."],
        ["06_build_customer_master_table.sql", "Left join all aggregates by SK_ID_CURR to create the final customer-level master table."],
        ["07_descriptive_statistics_and_segments.sql", "Produce descriptive statistics, bins, segments, and default-rate queries for analysis/dashboard."],
    ]
    story += [simple_table(sql_scope, [2.45 * inch, 4.65 * inch]), Spacer(1, 10)]
    clean = [
        ["Cleaning / feature rule", "Why it matters"],
        ["Missing-value flags", "Convert missingness into signal: flag = 1 if the original value was missing, 0 otherwise."],
        ["Median imputation", "Fill numeric missing values robustly so models can run without letting extreme values distort the replacement."],
        ["Safe division", "Avoid division-by-zero for ratios such as credit/income or utilization."],
        ["Special values", "Keep intentional categorical codes such as XNA/XAP where they represent business input rather than random missingness."],
        ["Historical aggregation", "Summarize many-to-one history into count, sum, mean, max, recency, rate, and coverage features."],
    ]
    story += [
        simple_table(clean, [2.15 * inch, 4.95 * inch]),
        Spacer(1, 8),
        conclusion(
            "In credit-risk data preparation, cleaning does not simply mean deleting missing values or forcing every unusual value into a generic replacement. "
            "Missingness, special categories, zero limits, and extreme financial ratios may contain business signal. "
            "The pipeline therefore uses missing flags, safe ratios, median imputation, and historical aggregation to keep the data usable for models while preserving credit-relevant information.",
            PURPLE,
        ),
        PageBreak(),
    ]

    add_dashboard_export_pages(story)

    story += [section_band("6. Dashboard Evidence: Five Core Risk Signals"), Spacer(1, 10)]
    story += [
        p(
            "This is the main DA evidence layer of the project. The actual interactive Power BI dashboard is included as <b>dashboard/dashboard.pbix</b>. "
            "The PDF shows enlarged dashboard evidence charts plus the supporting data tables so the report can be reviewed without opening Power BI.",
            "body",
        ),
        callout(
            "<b>How to read these dashboards:</b> the red dashed line is the portfolio baseline around 8.07%. A segment above that line has higher-than-average observed default risk and should receive more attention in review or monitoring.",
            BLUE,
        ),
        PageBreak(),
    ]

    story += [
        section_band("6.1 Dashboard Signal: Credit / Income Ratio"),
        Spacer(1, 8),
        p("<b>Decision meaning:</b> loan size must be evaluated relative to customer income. The 2x-4x credit/income band has the highest observed default rate in this view, so affordability should be reviewed with ratios, not raw loan amount alone.", "body"),
        fit_image("outputs/figures/step06_dashboard/04_default_rate_by_credit_income_ratio.png", 7.05 * inch, 3.0 * inch),
        Spacer(1, 6),
        simple_table(
            [
                ["Segment", "Default rate", "N applications", "Reading"],
                ["<=1", "6.4%", "16,174", "Below baseline; lower observed risk."],
                ["1-2", "7.7%", "60,164", "Near baseline; normal monitoring."],
                ["2-4", "8.9%", "113,561", "Highest band here; affordability pressure appears."],
                ["4-6", "8.1%", "62,887", "Around baseline; still needs ratio context."],
                ["6-10", "7.3%", "44,881", "Below baseline in this sample."],
                ["10+", "6.8%", "9,844", "Lower observed rate; possible selection/approval effect."],
            ],
            [1.2 * inch, 1.15 * inch, 1.25 * inch, 3.5 * inch],
        ),
        Spacer(1, 8),
        conclusion(
            "Credit / Income Ratio should not be interpreted as a simple linear rule. "
            "The 2x-4x segment has the highest observed default rate in this view, while higher bands do not continue increasing, likely because of approval selection effects. "
            "The practical conclusion is that this ratio is useful for affordability screening, but it should be combined with payment behavior, previous refusal, and card utilization before making a credit decision.",
            BLUE,
        ),
        PageBreak(),
    ]

    story += [
        section_band("6.2 Dashboard Signal: Annuity / Income Ratio"),
        Spacer(1, 8),
        p("<b>Decision meaning:</b> annuity/income measures monthly repayment burden. It is more operational than income alone because it asks whether the customer can carry the monthly payment after approval.", "body"),
        fit_image("outputs/figures/step06_dashboard/05_default_rate_by_annuity_income_ratio.png", 7.05 * inch, 3.0 * inch),
        Spacer(1, 6),
        simple_table(
            [
                ["Segment", "Default rate", "N applications", "Reading"],
                ["<=10%", "7.2%", "57,210", "Lower burden; below baseline."],
                ["10-20%", "8.1%", "145,231", "Around baseline; standard review."],
                ["20-30%", "8.8%", "73,742", "Highest observed rate; repayment pressure starts to show."],
                ["30-40%", "8.1%", "23,407", "Around baseline; check with other signals."],
                ["40-60%", "8.3%", "6,990", "Slightly above baseline; small segment."],
                ["60%+", "8.2%", "919", "Small group; do not over-read alone."],
            ],
            [1.2 * inch, 1.15 * inch, 1.25 * inch, 3.5 * inch],
        ),
        Spacer(1, 8),
        conclusion(
            "Annuity / Income Ratio is directly linked to monthly repayment capacity. "
            "The 20%-30% segment shows the highest default rate in this view, suggesting that repayment burden starts to matter once installments take a meaningful share of income. "
            "However, the pattern is not perfectly monotonic, so affordability should be treated as a necessary but not sufficient risk signal.",
            BLUE,
        ),
        PageBreak(),
    ]

    story += [
        section_band("6.3 Dashboard Signal: Previous Refusal Rate"),
        Spacer(1, 8),
        p("<b>Decision meaning:</b> previous refusal is a strong historical signal. As refusal rate rises, default rate rises from 7.1% to 17.8%, so repeated refusal history should trigger deeper document and affordability checks.", "body"),
        fit_image("outputs/figures/step06_dashboard/08_default_rate_by_previous_refusal_rate.png", 7.05 * inch, 3.0 * inch),
        Spacer(1, 6),
        simple_table(
            [
                ["Segment", "Default rate", "N applications", "Reading"],
                ["0", "7.1%", "190,763", "Lower than baseline; no prior refusal signal."],
                ["0-25%", "8.1%", "47,397", "Around baseline."],
                ["25-50%", "11.3%", "41,739", "Clear risk increase; review priority rises."],
                ["50-75%", "15.6%", "9,393", "High-risk history; strict review candidate."],
                ["75-100%", "17.8%", "1,765", "Very high risk; strong negative historical signal."],
                ["Missing", "6.0%", "16,454", "Missing history is not necessarily high risk here."],
            ],
            [1.2 * inch, 1.15 * inch, 1.25 * inch, 3.5 * inch],
        ),
        Spacer(1, 8),
        conclusion(
            "Previous refusal history is one of the clearest historical risk signals. "
            "Default rate rises from 7.1% with no refusal history to 17.8% in the 75%-100% refusal segment. "
            "This supports using refusal history as a manual review trigger, while avoiding automatic rejection because refusal reasons may differ across lenders and time periods.",
            BLUE,
        ),
        PageBreak(),
    ]

    story += [
        section_band("6.4 Dashboard Signal: Installment Late Payment Rate"),
        Spacer(1, 8),
        p("<b>Decision meaning:</b> late payment is a direct behavioral signal. The pattern is monotonic enough for business use: customers with 50%+ late payment rate show 16.4% default, roughly double the portfolio baseline.", "body"),
        fit_image("outputs/figures/step06_dashboard/09_default_rate_by_late_payment_rate.png", 7.05 * inch, 3.0 * inch),
        Spacer(1, 6),
        simple_table(
            [
                ["Segment", "Default rate", "N applications", "Reading"],
                ["No history", "6.0%", "15,868", "No installment record; lower observed risk."],
                ["0", "6.8%", "136,644", "No late payment; lower risk."],
                ["0-5%", "7.0%", "40,212", "Minor lateness; below baseline."],
                ["5-20%", "9.4%", "77,325", "Above baseline; review repayment behavior."],
                ["20-50%", "11.7%", "35,218", "Material late-payment risk."],
                ["50%+", "16.4%", "2,244", "Very high repayment stress; strict review signal."],
            ],
            [1.2 * inch, 1.15 * inch, 1.25 * inch, 3.5 * inch],
        ),
        Spacer(1, 8),
        conclusion(
            "Installment late payment is a strong behavioral signal because it reflects observed repayment behavior rather than declared profile information. "
            "Default rate increases from around 6.8%-7.0% for no or minimal late payment to 16.4% for the 50%+ late-payment segment. "
            "This makes late payment suitable for strict review prioritization when combined with affordability and bureau signals.",
            BLUE,
        ),
        PageBreak(),
    ]

    story += [
        section_band("6.5 Dashboard Signal: Credit Card Utilization"),
        Spacer(1, 8),
        p("<b>Decision meaning:</b> this is the clearest dashboard warning. Customers using more than 100% of their card limit show a 25.5% default rate, about 3.16x the 8.07% baseline.", "body"),
        fit_image("outputs/figures/step06_dashboard/11_default_rate_by_credit_card_utilization.png", 7.05 * inch, 3.0 * inch),
        Spacer(1, 6),
        simple_table(
            [
                ["Segment", "Default rate", "N applications", "Reading"],
                ["No card history", "7.8%", "220,606", "Near baseline; no card behavior signal."],
                ["<=0", "5.4%", "26,605", "Lowest observed risk in this view."],
                ["0-30%", "6.4%", "19,529", "Low utilization; below baseline."],
                ["30-70%", "9.5%", "24,183", "Above baseline; monitor utilization."],
                ["70-100%", "14.9%", "14,715", "High utilization; strict review candidate."],
                ["100%+", "25.5%", "1,004", "Severe credit line stress; strongest dashboard risk signal."],
                ["Missing", "11.9%", "869", "Small group but above baseline; verify data quality/history."],
            ],
            [1.2 * inch, 1.15 * inch, 1.25 * inch, 3.5 * inch],
        ),
        Spacer(1, 8),
        conclusion(
            "Credit card utilization is the strongest dashboard warning signal. "
            "Customers using more than 100% of their card limit show a 25.5% default rate, about 3.16x the 8.07% baseline. "
            "This indicates severe credit-line stress and should trigger stricter review, especially when combined with late payment or previous refusal history.",
            BLUE,
        ),
        PageBreak(),
    ]

    story += [section_band("6.6 Dashboard Conclusion"), Spacer(1, 10)]
    insight_table = [
        ["Dashboard signal", "Business interpretation"],
        ["Affordability burden", "Credit/income and annuity/income show why ratios are more decision-useful than raw money fields. A customer is risky not only because the loan is large, but because the obligation is large relative to repayment capacity."],
        ["Behavioral repayment stress", "Late payment and previous refusal provide evidence of realized friction, not just demographic/profile risk. These signals should receive high attention in manual review."],
        ["Credit line stress", "Credit card utilization above 100% is a direct operational warning. It indicates potential liquidity pressure and should feed stricter review rules."],
        ["Dashboard-to-model link", "These dashboard signals align with diagnostic and ML results, so the model is not a black box detached from the business story."],
    ]
    story += [
        simple_table(insight_table, [2.0 * inch, 5.1 * inch]),
        Spacer(1, 8),
        conclusion(
            "The dashboard evidence shows that default risk is not randomly distributed across the portfolio. "
            "It concentrates around interpretable credit-risk themes: repayment burden, previous refusal, late payment behavior, and credit-line stress. "
            "The dashboard therefore gives stakeholders a transparent reason to trust the data before moving into diagnostic modeling and ML scoring.",
            BLUE,
        ),
        PageBreak(),
    ]

    story += [section_band("7. Diagnostic Analytics: Explainable Logistic Regression"), Spacer(1, 10)]
    story += [
        p(
            "The diagnostic layer uses Logistic Regression to explain risk drivers in a controlled and business-readable way. It is not the final champion ML model; it is an interpretation layer that helps connect dashboard insight to model evidence.",
            "body",
        )
    ]
    diag = [
        ["Diagnostic design choice", "Rationale"],
        ["Why 28 variables, not all 271?", "Diagnostic analysis prioritizes representative, interpretable variables from the main risk themes. The full 271-feature set is too wide and partly redundant for coefficient interpretation."],
        ["VIF check", "Used to identify overlapping variables before reading coefficients and Odds Ratios. High-VIF raw money variables are not interpreted separately when ratios/bins tell the business story more clearly."],
        ["Odds Ratio", "Exp(coefficient). OR > 1 means higher odds of default relative to the reference group; OR < 1 means lower odds."],
        ["Main diagnostic result", "Logistic model achieved ROC-AUC about 0.685, Lift@10 2.53x, and top 10% default rate 20.42% versus 8.07% baseline."],
    ]
    story += [simple_table(diag, [2.0 * inch, 5.1 * inch]), Spacer(1, 10)]
    story += [
        fit_image("outputs/figures/step06_dashboard/06_step6_charts/roc_curve_main_diagnostic_models.png", 5.75 * inch, 2.55 * inch),
        Spacer(1, 8),
        p("<b>Interpretation:</b> diagnostic modeling proves that the selected risk themes have signal, while keeping the explanation understandable for credit stakeholders.", "body"),
        Spacer(1, 8),
        conclusion(
            "The diagnostic model confirms that the dashboard-selected risk themes still carry signal inside a controlled Logistic Regression framework. "
            "It uses 28 representative variables instead of all 271 features because the purpose is explanation, not maximum prediction. "
            "Lift@10 of 2.53x and a top-10% default rate of 20.42% show that the selected variables meaningfully separate higher-risk customers from the 8.07% baseline.",
            PURPLE,
        ),
        PageBreak(),
    ]

    story += [section_band("8. ML Benchmarking and Champion Model"), Spacer(1, 10)]
    story += [
        p(
            "The ML layer focuses on ranking customers by default risk. Because TARGET=1 is only around 8-9% of observations, the project uses ROC-AUC, PR-AUC, Lift, and capture rate instead of relying on accuracy.",
            "body",
        )
    ]
    split = [
        ["Dataset split / metric", "Meaning"],
        ["Fit train: 196,805 rows, default 7.75%", "Main training partition used to fit model parameters."],
        ["Calibration: 49,202 rows, default 7.75%", "Held-out partition used to calibrate/compare scores and avoid overconfident probability interpretation."],
        ["Validation: 61,504 rows, default 9.37%", "Final holdout used to evaluate ranking quality and business risk bands."],
        ["OOF AUC: 0.7978", "Out-of-fold validation estimate from cross-validation; helps check leak-free performance."],
        ["Validation AUC: 0.7907", "Final holdout result; close to OOF, indicating stable generalization."],
    ]
    story += [simple_table(split, [2.0 * inch, 5.1 * inch]), Spacer(1, 10)]
    model_metrics = [
        metric_card("0.7907", "Validation ROC-AUC", BLUE),
        metric_card("0.3127", "Validation PR-AUC", RED),
        metric_card("3.66x", "Lift@10", GREEN),
        metric_card("LightGBM", "champion model", PURPLE),
        metric_card("0.500", "random ROC-AUC baseline", AMBER),
    ]
    story.append(Table([model_metrics], colWidths=[1.42 * inch] * 5))
    story += [Spacer(1, 10)]
    story += [
        fit_image("outputs/figures/step08_ml/08_step8_figures/v3_advanced_cell14_img01.png", 6.65 * inch, 2.25 * inch),
        Spacer(1, 8),
        conclusion(
            "The ML layer is used for risk ranking, not for unconditional automatic decisions. "
            "Validation ROC-AUC of 0.7907, PR-AUC of 0.3127, and Lift@10 of 3.66x show that the model is able to concentrate default cases into the highest-score region. "
            "This is more meaningful than accuracy because default is a minority event and the business value comes from prioritizing the right cases for review.",
            PURPLE,
        ),
        PageBreak(),
    ]

    story += [section_band("9. Turning Scores into Business Action"), Spacer(1, 10)]
    story += [
        p(
            "The most useful output is not the score itself, but the decision system built around the score. "
            "A practical analytics view is to avoid treating ML as an automatic approval/rejection machine. The model should help the business decide where to spend review capacity, what evidence to check, and which risk conditions deserve stricter policy control.",
            "body",
        )
    ]
    band = [
        ["Band", "Operational use", "Observed validation risk"],
        ["Green / fast-track candidate", "Low-risk applications can be processed faster if they also pass hard policy rules. The goal is shorter turnaround time, not blind approval.", "About 4.27% default rate"],
        ["Amber / manual review", "Middle-risk applications should receive standard analyst review, with attention to affordability ratios and recent repayment behavior.", "About 15.92% default rate"],
        ["Red / strict review or reject candidate", "Highest-risk applications need stronger controls: document verification, affordability review, bureau/payment-history checks, lower limit, or policy rejection if multiple red flags align.", "About 32.50% default rate"],
    ]
    story += [simple_table(band, [1.6 * inch, 3.55 * inch, 1.95 * inch]), Spacer(1, 10)]
    story += [
        fit_image("outputs/figures/step08_ml/08_step8_figures/raw_benchmark_cell11_img01.png", 5.45 * inch, 2.05 * inch),
        Spacer(1, 8),
        callout(
            "<b>Business impact:</b> reviewing the top 29.6% highest-risk applications captures about 67.9% of default cases, improving targeting efficiency by about 2.27x versus random review.",
            GREEN,
        ),
        PageBreak(),
    ]

    story += [section_band("10. Decision Recommendation and Business Conclusion"), Spacer(1, 10)]
    story += [
        p(
            "The analytical conclusion is that default risk is not random across the portfolio. It concentrates in measurable and explainable groups: customers with high repayment burden, weak historical application outcomes, late payment behavior, and credit line stress. "
            "Therefore, the business should not allocate review effort evenly across all applications. Review should be risk-weighted.",
            "body",
        ),
        p(
            "The dashboard gives the business rule candidates, while the ML score ranks customers across many signals at once. The best operating design is to combine both: use dashboard-derived rules to explain why a case is risky, and use the model score to prioritize which cases should be reviewed first.",
            "body",
        ),
    ]
    decision_actions = [
        ["Decision question", "Recommended action from the analysis"],
        ["Which cases can move faster?", "Applications in the green score band, with no hard policy breaches and no severe dashboard red flags, can be routed to faster processing. This improves operational efficiency without claiming that the model alone approves the customer."],
        ["Which cases need standard review?", "Amber-band cases should receive normal analyst review. Analysts should focus on affordability ratios, previous refusals, and payment-history behavior because these signals repeatedly appear in dashboard, diagnostic, and ML layers."],
        ["Which cases need strict control?", "Red-band cases should receive enhanced verification. A customer is especially high priority if several red flags align: credit card utilization above 100%, high previous refusal rate, high late-payment rate, and elevated repayment burden."],
        ["What is the measurable business value?", "Instead of randomly reviewing applications, reviewing the top 29.6% highest-risk group captures about 67.9% of default cases. This means review resources are concentrated where risk is most likely to appear."],
        ["What should not be automated?", "The score should not become an automatic rejection rule. Credit decisions require policy, compliance, explainability, and human judgment, especially because external-source features are predictive but black-box."],
    ]
    story += [simple_table(decision_actions, [1.95 * inch, 5.15 * inch]), Spacer(1, 8)]
    story += [
        conclusion(
            "The strongest business value comes from changing the review process. "
            "Instead of treating all applications equally, the model and dashboard together create a risk-weighted workflow: fast-track low-risk applications, review medium-risk applications with clear evidence, and apply strict controls to concentrated high-risk applications. "
            "This turns analytics into an operating decision, not just a report metric.",
            GREEN,
        ),
        PageBreak(),
    ]

    story += [section_band("11. Decision Value: Policy and Monitoring Design"), Spacer(1, 10)]
    policy = [
        ["Area", "How the project should be used in practice"],
        ["Credit policy", "Use dashboard thresholds as policy discussion inputs: card utilization above 100%, high refusal history, and high late-payment rate should trigger stricter review rather than simple score-only decisions."],
        ["Operations", "Use the risk score to queue applications. Analysts should start with the highest-risk cases because the top review group captures a disproportionate share of defaults."],
        ["Portfolio monitoring", "Track default rate by risk band over time. If green-band risk rises or red-band capture falls, investigate data drift, policy changes, or model decay."],
        ["Customer treatment", "Avoid a one-size-fits-all rejection logic. Some high-risk cases may need lower exposure, more documents, or pricing/limit adjustment rather than outright rejection."],
        ["Governance", "Monitor sensitive/proxy variables and black-box external scores. A model can have good AUC and still require fairness, compliance, and business review before deployment."],
    ]
    story += [simple_table(policy, [1.55 * inch, 5.55 * inch]), Spacer(1, 10)]
    story += [
        p(
            "<b>Final interpretation:</b> the strongest value of this project is not simply building a predictive model. The value is turning fragmented credit history into a master analytical table, converting that table into dashboard insights, validating the signals with diagnostic modeling, and finally translating the ML score into a controlled business workflow.",
            "body",
        ),
        p(
            "This demonstrates the full chain from data preparation to decision support: clean data, define risk signals, quantify lift, communicate trade-offs, and propose a process that a credit team can actually use.",
            "body",
        ),
        Spacer(1, 8),
        conclusion(
            "The recommended operating model is to use dashboard rules and model scores together. "
            "Dashboard signals explain why an application is risky, while the ML score prioritizes which applications should be reviewed first. "
            "This combination builds confidence in the data because business users can trace the score back to understandable credit-risk drivers.",
            GREEN,
        ),
        PageBreak(),
    ]

    story += [section_band("12. Explainability and Governance"), Spacer(1, 10)]
    story += [
        p(
            "SHAP was used to explain which features drove model output. This is important because a high-performing credit model still needs transparency, monitoring, and governance before it can support decisions.",
            "body",
        ),
        fit_image("outputs/figures/step08_ml/08_step8_figures/v3_advanced_cell14_img05.png", 4.65 * inch, 4.85 * inch),
        Spacer(1, 8),
    ]
    gov = [
        ["Governance area", "How it was handled"],
        ["Black-box external scores", "EXT_SOURCE variables are strong predictors, but they require explanation and monitoring because their upstream calculation is unknown."],
        ["Sensitive/proxy bias", "Occupation, organization, and gender-related variables were tested through sensitivity and fairness-gap checks."],
        ["AUC sensitivity", "Removing occupation/organization/gender barely reduced AUC, showing these variables are not essential for predictive power."],
        ["Human-in-the-loop", "Scores support prioritization; final approval/rejection must still follow policy, compliance, and analyst judgment."],
    ]
    story += [
        simple_table(gov, [1.8 * inch, 5.3 * inch]),
        Spacer(1, 8),
        conclusion(
            "In credit risk, a model with good AUC is still incomplete without governance. "
            "Strong predictors such as external scores may be black-box, and profile-related variables may create proxy-bias concerns. "
            "The correct deployment posture is therefore human-in-the-loop: use the score to prioritize review, monitor fairness and drift, and keep final approval decisions under policy and compliance control.",
            RED,
        ),
        PageBreak(),
    ]

    story += [section_band("13. Evidence, Reproducibility, and Handover Notes"), Spacer(1, 10)]
    evidence = [
        ["Repository artifact", "What it proves"],
        ["README.md", "Project overview, results, run instructions, and repo structure."],
        ["sql/", "SQL ETL implementation for cleaning, feature engineering, historical aggregation, and master table."],
        ["src/", "Python scripts for data understanding, cleaning, descriptive analytics, master build, and diagnostic analytics."],
        ["notebooks/step08_machine_learning_governance.ipynb", "ML benchmarking, scoring, risk bands, SHAP, and governance checks."],
        ["dashboard/dashboard.pbix", "Main interactive Power BI dashboard file for portfolio and segment monitoring."],
        ["dashboard/dashboard.pdf", "Power BI dashboard export inserted directly into this portfolio PDF as dashboard pages."],
        ["data/processed/final_customer_analysis_train.csv.gz", "Final labeled customer-level processed data used for analytics/modeling evidence."],
        ["outputs/tables/ and outputs/figures/", "Result tables and charts supporting the presentation and portfolio PDF."],
    ]
    story += [simple_table(evidence, [2.65 * inch, 4.45 * inch]), Spacer(1, 10)]
    story += [
        KeepTogether(
            [
                p("<b>Project explanation summary:</b>", "h2"),
                bullet("I used SQL for the data engineering logic: cleaning flags, missing flags, feature engineering, joins, and customer-level aggregation."),
                bullet("I used Python for analytics and modeling because ML evaluation, Logistic Regression, LightGBM, SHAP, and validation workflows are more suitable there."),
                bullet("I used Power BI to convert the analytical table into stakeholder-facing monitoring views."),
                bullet("The main business value is not just AUC; it is review prioritization: the top 29.6% risk group captures about 67.9% of default cases."),
            ]
        ),
        Spacer(1, 8),
        conclusion(
            "The project demonstrates the full analytics chain from data preparation to business decision support. "
            "The central finding is that default risk is concentrated in measurable groups, especially customers with repayment stress, previous refusal history, late-payment behavior, and credit-line overuse. "
            "The main value is therefore not only model performance, but the ability to help the business review the right applications first and justify that review with transparent data evidence.",
            PURPLE,
        ),
        Spacer(1, 8),
        callout(
            "<b>Project summary:</b> Built a credit risk analytics and ML scoring pipeline on 307K+ labeled applications, created a 271-feature customer-level master table, identified high-risk borrower segments, and developed risk bands that improved review targeting efficiency by 2.27x versus random review.",
            PURPLE,
        ),
    ]

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUT)


if __name__ == "__main__":
    build()
