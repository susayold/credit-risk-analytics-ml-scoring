import {
  ArrowDown,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  Code2,
  Database,
  Download,
  GitBranch,
  Layers3,
  ShieldCheck,
} from "lucide-react";
import Image from "next/image";
import { DashboardGallery } from "./DashboardGallery";

const findings = [
  {
    number: "01",
    title: "Payment distress is the clearest actionable warning",
    body: "Customers using more than 100% of their card limit recorded a 25.50% default rate, 3.16x the 8.07% portfolio baseline. Late-payment and underpayment rates above 30% also rose to 13.35% and 12.34%. These are behavioral signals, so they are more operationally useful than demographics alone.",
    metric: "25.50%",
    label: "default at >100% card utilization",
    tone: "danger",
  },
  {
    number: "02",
    title: "External credit history reveals severe hidden exposure",
    body: "Borrowers with at least two overdue bureau loans reached a 36.80% default rate. Customers with more than 50% of previous applications refused reached 15.93%. The evidence supports combining the current application with historical obligations before making a credit decision.",
    metric: "36.80%",
    label: "default with 2+ overdue bureau loans",
    tone: "danger",
  },
  {
    number: "03",
    title: "Affordability ratios need context, not a single cutoff",
    body: "Credit-to-income and annuity-to-income relationships are not perfectly linear. A large loan is not automatically high risk when income and repayment behavior support it. Ratios work best as screening features alongside payment and bureau signals.",
    metric: "271",
    label: "features in the customer master table",
    tone: "neutral",
  },
  {
    number: "04",
    title: "Risk bands create a practical review queue",
    body: "The dashboard's rule-based segmentation rises from 4.66% default in the low-risk group to 15.06% in the very-high-risk group. This ranking is suitable for prioritizing review capacity, but it should not become an automatic rejection rule.",
    metric: "4.66 → 15.06%",
    label: "default from low to very-high risk",
    tone: "positive",
  },
];

const pipeline = [
  { step: "01", title: "Understand", copy: "Define TARGET=1, data grain and the cost of approval errors.", icon: Database },
  { step: "02", title: "Prepare", copy: "Clean special values, preserve missingness and engineer application flags.", icon: Code2 },
  { step: "03", title: "Aggregate", copy: "Use SQL joins and customer-level aggregation across historical tables.", icon: Layers3 },
  { step: "04", title: "Explain", copy: "Build Power BI views, descriptive statistics and diagnostic models.", icon: BarChart3 },
  { step: "05", title: "Prioritize", copy: "Benchmark ML models and translate scores into review actions.", icon: ShieldCheck },
];

export default function Home() {
  return (
    <main>
      <nav className="site-nav" aria-label="Primary navigation">
        <a className="brand" href="#top"><span>CR</span><strong>Credit Risk Case Study</strong></a>
        <div className="nav-links">
          <a href="#dashboard">Dashboard</a>
          <a href="#findings">Findings</a>
          <a href="#model">Model</a>
          <a href="#evidence">Evidence</a>
        </div>
        <a className="icon-link" href="https://github.com/susayold/credit-risk-analytics-ml-scoring" target="_blank" rel="noreferrer" aria-label="Open GitHub repository"><GitBranch size={20} /></a>
      </nav>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">Python · SQL · Power BI · Machine Learning</p>
          <h1>Credit Risk Analytics &amp; ML Scoring Pipeline</h1>
          <p className="hero-summary">An end-to-end case study that turns 307K labeled loan applications and multi-table credit history into a 271-feature customer view, an interactive Power BI decision layer and an ML-based review queue.</p>
          <div className="hero-actions">
            <a className="button primary" href="#dashboard"><BarChart3 size={18} /> Explore the dashboard</a>
            <a className="button secondary" href="/credit-risk-power-bi-dashboard.pdf" target="_blank"><Download size={18} /> Dashboard PDF</a>
          </div>
        </div>
        <div className="hero-facts" aria-label="Project highlights">
          <div><strong>307,511</strong><span>labeled applications</span></div>
          <div><strong>271</strong><span>customer-level features</span></div>
          <div><strong>8.07%</strong><span>portfolio default baseline</span></div>
          <div><strong>0.7907</strong><span>LightGBM ROC-AUC</span></div>
        </div>
        <a className="hero-dashboard-preview" href="#dashboard" aria-label="Explore the complete Power BI dashboard">
          <Image src="/dashboard/dashboard_page_01.png" alt="Power BI credit risk portfolio overview" width={1600} height={900} priority unoptimized />
          <span><BarChart3 size={17} /> Power BI Portfolio Overview <ArrowDown size={17} /></span>
        </a>
      </section>

      <section className="section dashboard-section" id="dashboard">
        <div className="section-heading split-heading">
          <div>
            <p className="eyebrow">Power BI · six decision views</p>
            <h2>The dashboard is the operational front door</h2>
          </div>
          <p>Each page answers a different credit question, from portfolio health and borrower affordability to historical repayment behavior and review priority.</p>
        </div>
        <DashboardGallery />
      </section>

      <section className="section findings-section" id="findings">
        <div className="section-heading">
          <p className="eyebrow">What the data says</p>
          <h2>Business conclusions, not just charts</h2>
          <p>The analysis separates signals that are useful for operational action from variables that are only descriptive or require governance.</p>
        </div>
        <div className="findings-list">
          {findings.map((finding) => (
            <article className="finding" key={finding.number}>
              <span className="finding-number">{finding.number}</span>
              <div className="finding-copy"><h3>{finding.title}</h3><p>{finding.body}</p></div>
              <div className={`finding-metric ${finding.tone}`}><strong>{finding.metric}</strong><span>{finding.label}</span></div>
            </article>
          ))}
        </div>
      </section>

      <section className="section model-section" id="model">
        <div className="section-heading split-heading">
          <div><p className="eyebrow">Model performance</p><h2>From risk score to a focused review queue</h2></div>
          <p>LightGBM was selected as the practical champion because the weighted ensemble produced no meaningful AUC improvement while adding complexity.</p>
        </div>
        <div className="model-grid">
          <div className="model-scoreboard">
            <div><span>ROC-AUC</span><strong>0.7907</strong><small>Ranking power across thresholds</small></div>
            <div><span>PR-AUC</span><strong>0.3127</strong><small>Focus on the minority default class</small></div>
            <div><span>KS</span><strong>0.437</strong><small>Separation between good and bad borrowers</small></div>
            <div><span>Lift@10</span><strong>3.66x</strong><small>Risk concentration in the top decile</small></div>
          </div>
          <div className="review-story">
            <p className="eyebrow">Validation decision</p>
            <div className="review-stat"><strong>Top 30%</strong><span>highest model scores selected for review</span></div>
            <div className="capture-bar" aria-label="69.1 percent of defaults captured"><span style={{ width: "69.1%" }} /></div>
            <div className="capture-label"><strong>69.1%</strong><span>of validation defaults captured</span></div>
            <p>This concentrates default cases at <strong>2.30x</strong> the rate expected from random review. The model ranks review priority; it does not make an automatic lending decision.</p>
          </div>
        </div>
      </section>

      <section className="section pipeline-section" id="pipeline">
        <div className="section-heading"><p className="eyebrow">End-to-end ownership</p><h2>How the evidence was built</h2></div>
        <div className="pipeline-grid">
          {pipeline.map(({ step, title, copy, icon: Icon }) => (
            <div className="pipeline-item" key={step}><div className="pipeline-icon"><Icon size={20} /></div><span>{step}</span><h3>{title}</h3><p>{copy}</p></div>
          ))}
        </div>
        <div className="tech-strip">
          <span>SQL</span><p>Cleaning flags, feature engineering, multi-table joins and customer-level aggregation.</p>
          <span>Power BI</span><p>Six-page portfolio analysis with drillable borrower, loan, history and behavior views.</p>
          <span>Python</span><p>Statistical analysis, diagnostic Logistic Regression, LightGBM benchmarking, SHAP and governance checks.</p>
        </div>
      </section>

      <section className="section governance-section">
        <div className="governance-copy">
          <p className="eyebrow">Explainability &amp; governance</p>
          <h2>A strong score is not enough for a credit decision</h2>
          <p>SHAP identifies external credit scores and affordability ratios as major model signals. However, external scores are partially black-box, while occupation and organization can become proxy variables for protected characteristics. The recommended design is human-in-the-loop: use the score to prioritize review, monitor group gaps and require policy-based final decisions.</p>
          <ul>
            <li><CheckCircle2 size={18} /> External score and affordability dominate global importance.</li>
            <li><CheckCircle2 size={18} /> Removing sensitive features barely changes AUC, but fairness gaps can remain through proxies.</li>
            <li><CheckCircle2 size={18} /> Demographics are used for monitoring and context, not standalone rejection rules.</li>
          </ul>
        </div>
        <div className="shap-list" aria-label="Top SHAP signals">
          <p>Global SHAP importance</p>
          {[
            ["External score mean", 0.129, "100%"],
            ["Annuity / credit ratio", 0.086, "67%"],
            ["Organization type", 0.076, "59%"],
            ["External source 2", 0.069, "53%"],
            ["Amount annuity", 0.058, "45%"],
            ["External source 3", 0.057, "44%"],
          ].map(([name, value, width]) => (
            <div className="shap-row" key={String(name)}><div><span>{name}</span><strong>{Number(value).toFixed(3)}</strong></div><i><b style={{ width: String(width) }} /></i></div>
          ))}
        </div>
      </section>

      <section className="section evidence-section" id="evidence">
        <div className="section-heading split-heading">
          <div><p className="eyebrow">Reproducible evidence</p><h2>Inspect the work behind the claims</h2></div>
          <p>The repository is organized by purpose so reviewers can move directly from code to output without searching through draft files.</p>
        </div>
        <div className="evidence-links">
          <a href="https://github.com/susayold/credit-risk-analytics-ml-scoring/tree/main/sql" target="_blank" rel="noreferrer"><Database size={22} /><div><strong>SQL ETL</strong><span>Seven scripts from base cleaning to customer master and segment analysis.</span></div><ArrowUpRight size={18} /></a>
          <a href="https://github.com/susayold/credit-risk-analytics-ml-scoring/tree/main/src" target="_blank" rel="noreferrer"><Code2 size={22} /><div><strong>Python pipeline</strong><span>Descriptive, diagnostic and modeling code with generated result tables.</span></div><ArrowUpRight size={18} /></a>
          <a href="/credit-risk-power-bi-dashboard.pdf" target="_blank"><BarChart3 size={22} /><div><strong>Power BI export</strong><span>Open the complete six-page dashboard as a PDF.</span></div><ArrowUpRight size={18} /></a>
          <a href="https://github.com/susayold/credit-risk-analytics-ml-scoring/tree/main/outputs" target="_blank" rel="noreferrer"><ShieldCheck size={22} /><div><strong>Result evidence</strong><span>Tables and figures supporting every reported metric.</span></div><ArrowUpRight size={18} /></a>
        </div>
      </section>

      <footer>
        <div><strong>Credit Risk Analytics &amp; ML Scoring</strong><p>Portfolio case study built for transparent, risk-aware decision support.</p></div>
        <a href="https://github.com/susayold/credit-risk-analytics-ml-scoring" target="_blank" rel="noreferrer"><GitBranch size={18} /> View repository</a>
      </footer>
    </main>
  );
}
