import React, { useRef } from "react";
import Hero from "./components/Hero.jsx";
import DetectionUI from "./components/DetectionUI.jsx";
import "./ui.css";

export default function App() {
  const detectRef = useRef(null);

  const scrollToDetect = () => {
    detectRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="app">
      <Hero onStart={scrollToDetect} />
      <section ref={detectRef} id="detect" className="tool-section">
        <DetectionUI />
      </section>
      <footer className="ftr">
        © {new Date().getFullYear()} Defective Exhibit ID • Gallery Edition
      </footer>
    </div>
  );
}
