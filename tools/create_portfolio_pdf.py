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
OUT = ROOT / "reports" / "portfolio_credit_risk_analytics_ml_scoring_detailed.pdf"

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
    story += [Spacer(1, 6), simple_table(deliverables, [1.55 * inch, 5.55 * inch]), PageBreak()]


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
        callout(
            "<b>Portfolio message:</b> I framed the work as a business decision support system: use analytics to decide where review resources should go first.",
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
        PageBreak(),
    ]

    story += [section_band("4. SQL ETL and Cleaning Logic"), Spacer(1, 10)]
    story += [
        p(
            "The original project was executed mainly with Python/pandas. I added a SQL ETL version to document how the data engineering layer can be implemented with SQL for interview and production-style explanation.",
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
    story += [simple_table(clean, [2.15 * inch, 4.95 * inch]), PageBreak()]

    story += [section_band("5. Descriptive Analytics and Dashboard Insights"), Spacer(1, 10)]
    story += [
        p(
            "Before modeling, the project uses descriptive analytics to identify business-readable risk patterns. These insights are easier to explain to stakeholders than raw model coefficients.",
            "body",
        ),
        fit_image("outputs/figures/step06_dashboard/11_default_rate_by_credit_card_utilization.png", 6.7 * inch, 2.8 * inch),
        Spacer(1, 8),
    ]
    insight_table = [
        ["Finding", "Business interpretation"],
        ["Credit card utilization >100%: 25.5% default rate", "Strong stress signal; customers using beyond the limit should be reviewed more tightly."],
        ["3.16x risk versus 8.07% baseline", "The segment is not just slightly risky; it is several times riskier than the portfolio average."],
        ["Payment behavior matters", "Late payment, underpayment, POS DPD, and previous refusal patterns are consistent risk signals."],
        ["External score variables are powerful but black-box", "EXT_SOURCE fields help prediction, but governance is needed because their upstream logic is not transparent."],
    ]
    story += [simple_table(insight_table, [2.35 * inch, 4.75 * inch]), Spacer(1, 8)]
    story += [
        callout(
            "<b>Dashboard role:</b> Power BI converts the model and descriptive outputs into monitoring views by borrower profile, credit burden, bureau history, repayment behavior, and card utilization.",
            BLUE,
        ),
        PageBreak(),
    ]

    story += [section_band("6. Diagnostic Analytics: Explainable Logistic Regression"), Spacer(1, 10)]
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
        PageBreak(),
    ]

    story += [section_band("7. ML Benchmarking and Champion Model"), Spacer(1, 10)]
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
        PageBreak(),
    ]

    story += [section_band("8. Turning Scores into Business Action"), Spacer(1, 10)]
    story += [
        p(
            "The most useful business output is the three-way decision framework. Instead of asking the model to make a yes/no decision, the score is translated into operational review bands.",
            "body",
        )
    ]
    band = [
        ["Band", "Operational use", "Observed validation risk"],
        ["Green / auto-approve candidate", "Low-risk applications can be processed faster subject to policy rules.", "About 4.27% default rate"],
        ["Amber / manual review", "Middle-risk applications should receive normal analyst review.", "About 15.92% default rate"],
        ["Red / strict review or reject candidate", "Highest-risk applications need stronger controls, document checks, or conservative policy treatment.", "About 32.50% default rate"],
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

    story += [section_band("9. Explainability and Governance"), Spacer(1, 10)]
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
    story += [simple_table(gov, [1.8 * inch, 5.3 * inch]), PageBreak()]

    story += [section_band("10. Evidence, Reproducibility, and Interview Talking Points"), Spacer(1, 10)]
    evidence = [
        ["Repository artifact", "What it proves"],
        ["README.md", "Project overview, results, run instructions, and repo structure."],
        ["sql/", "SQL ETL implementation for cleaning, feature engineering, historical aggregation, and master table."],
        ["src/", "Python scripts for data understanding, cleaning, descriptive analytics, master build, and diagnostic analytics."],
        ["notebooks/step08_machine_learning_governance.ipynb", "ML benchmarking, scoring, risk bands, SHAP, and governance checks."],
        ["dashboard/credit_risk_dashboard.pbix", "Power BI dashboard for portfolio and segment monitoring."],
        ["data/processed/final_customer_analysis_train.csv.gz", "Final labeled customer-level processed data used for analytics/modeling evidence."],
        ["outputs/tables/ and outputs/figures/", "Result tables and charts supporting the presentation and portfolio PDF."],
    ]
    story += [simple_table(evidence, [2.65 * inch, 4.45 * inch]), Spacer(1, 10)]
    story += [
        KeepTogether(
            [
                p("<b>How I would explain this in an interview:</b>", "h2"),
                bullet("I used SQL for the data engineering logic: cleaning flags, missing flags, feature engineering, joins, and customer-level aggregation."),
                bullet("I used Python for analytics and modeling because ML evaluation, Logistic Regression, LightGBM, SHAP, and validation workflows are more suitable there."),
                bullet("I used Power BI to convert the analytical table into stakeholder-facing monitoring views."),
                bullet("The main business value is not just AUC; it is review prioritization: the top 29.6% risk group captures about 67.9% of default cases."),
            ]
        ),
        Spacer(1, 8),
        callout(
            "<b>CV-ready summary:</b> Built a credit risk analytics and ML scoring pipeline on 307K+ labeled applications, created a 271-feature customer-level master table, identified high-risk borrower segments, and developed risk bands that improved review targeting efficiency by 2.27x versus random review.",
            PURPLE,
        ),
    ]

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUT)


if __name__ == "__main__":
    build()
