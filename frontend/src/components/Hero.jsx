import React from "react";
import "./hero.css";

export default function Hero({ onStart }) {
  return (
    <section className="hero">
      <div className="bg-grad" />
      <div className="blob blob-a" />
      <div className="blob blob-b" />
      <div className="grain" />

      <nav className="nav glass">
        <div className="brand">
          <div className="logo" />
          <span className="brand-name">Image Defect Detection</span>
        </div>
        <div className="nav-right">
          <span className="chip">Mask R-CNN</span>
          <span className="chip">FastAPI</span>
          <span className="chip">React</span>
        </div>
      </nav>

      <div className="hero-inner">
        <h1 className="title">
          AI-Based <span className="underline">Defective Exhibit</span> Identification
        </h1>
        <p className="subtitle">
          Detect cracks, scratches, stains, and surface anomalies on artwork and industrial items.
          Fine-tuned Mask R-CNN with a sleek, gallery-style interface.
        </p>

        <div className="cta">
          <button className="btn primary" onClick={onStart}>Start Detection</button>
        </div>

        <div className="stats">
          <div>Backbone: <b>ResNet-50 FPN</b></div>
          <div>Interaction: <b>Zoom & Pan</b></div>
          <div>Drawer: <b>Defects Panel</b></div>
        </div>
      </div>
    </section>
  );
}
