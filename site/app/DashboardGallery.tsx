"use client";

import { ChevronLeft, ChevronRight, Maximize2, X } from "lucide-react";
import Image from "next/image";
import { useEffect, useState } from "react";

const dashboards = [
  {
    title: "Portfolio Overview",
    caption: "Baseline risk, portfolio size and the first view of default concentration.",
    src: "/dashboard/dashboard_page_01.png?v=2341efb5",
  },
  {
    title: "Customer Profile",
    caption: "Borrower demographics and customer composition used for descriptive context.",
    src: "/dashboard/dashboard_page_02.png?v=2341efb5",
  },
  {
    title: "Loan & Affordability",
    caption: "Credit size, income and repayment-burden ratios by risk segment.",
    src: "/dashboard/dashboard_page_03.png?v=2341efb5",
  },
  {
    title: "Credit History",
    caption: "Bureau exposure, overdue accounts and prior application outcomes.",
    src: "/dashboard/dashboard_page_04.png?v=2341efb5",
  },
  {
    title: "Payment Behavior",
    caption: "Late payment, underpayment and revolving credit utilization signals.",
    src: "/dashboard/dashboard_page_05.png?v=2341efb5",
  },
  {
    title: "Risk Segmentation",
    caption: "Rule-based risk bands translated into review priorities for operations.",
    src: "/dashboard/dashboard_page_06.png?v=2341efb5",
  },
];

export function DashboardGallery() {
  const [active, setActive] = useState(0);
  const [expanded, setExpanded] = useState(false);

  const move = (direction: number) => {
    setActive((current) => (current + direction + dashboards.length) % dashboards.length);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!expanded) return;
      if (event.key === "Escape") setExpanded(false);
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [expanded]);

  const selected = dashboards[active];

  return (
    <div className="gallery-shell">
      <div className="gallery-stage">
        <button className="gallery-arrow gallery-arrow-left" onClick={() => move(-1)} aria-label="Previous dashboard">
          <ChevronLeft size={22} />
        </button>
        <button className="gallery-image-button" onClick={() => setExpanded(true)} aria-label={`Open ${selected.title} at full size`}>
          <Image src={selected.src} alt={`${selected.title} Power BI dashboard`} width={1600} height={900} priority unoptimized />
          <span className="expand-label"><Maximize2 size={16} /> View full size</span>
        </button>
        <button className="gallery-arrow gallery-arrow-right" onClick={() => move(1)} aria-label="Next dashboard">
          <ChevronRight size={22} />
        </button>
      </div>

      <div className="gallery-meta" aria-live="polite">
        <div>
          <span className="gallery-index">0{active + 1} / 0{dashboards.length}</span>
          <h3>{selected.title}</h3>
          <p>{selected.caption}</p>
        </div>
        <div className="gallery-dots" role="tablist" aria-label="Dashboard pages">
          {dashboards.map((dashboard, index) => (
            <button
              key={dashboard.title}
              className={index === active ? "active" : ""}
              onClick={() => setActive(index)}
              aria-label={`Show ${dashboard.title}`}
              aria-selected={index === active}
              role="tab"
            />
          ))}
        </div>
      </div>

      <div className="gallery-thumbnails">
        {dashboards.map((dashboard, index) => (
          <button key={dashboard.title} onClick={() => setActive(index)} className={index === active ? "active" : ""}>
            <Image src={dashboard.src} alt="" width={320} height={180} unoptimized />
            <span>{dashboard.title}</span>
          </button>
        ))}
      </div>

      {expanded && (
        <div className="lightbox" role="dialog" aria-modal="true" aria-label={selected.title} onClick={() => setExpanded(false)}>
          <button className="lightbox-close" onClick={() => setExpanded(false)} aria-label="Close full-size dashboard"><X size={24} /></button>
          <button className="lightbox-arrow lightbox-left" onClick={(event) => { event.stopPropagation(); move(-1); }} aria-label="Previous dashboard"><ChevronLeft size={28} /></button>
          <Image src={selected.src} alt={`${selected.title} Power BI dashboard full size`} width={1600} height={900} unoptimized onClick={(event) => event.stopPropagation()} />
          <button className="lightbox-arrow lightbox-right" onClick={(event) => { event.stopPropagation(); move(1); }} aria-label="Next dashboard"><ChevronRight size={28} /></button>
        </div>
      )}
    </div>
  );
}
