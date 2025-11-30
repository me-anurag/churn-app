import React, { useState } from "react";
import "./styles.css";
import Navbar from "./components/Navbar";
import InfoSection from "./components/InfoSection";
import UploadSection from "./components/UploadSection";
import ResultsSection from "./components/ResultsSection";
import VisualizationSection from "./components/VisualizationSection";

function App() {
  const [results, setResults] = useState(null);

  return (
    <div>
      <Navbar />

      <div className="main-container">
        <InfoSection />
        <UploadSection setResults={setResults} />

        {results && <ResultsSection results={results} />}

        {/* Show visualizations AFTER the model produces results */}
        {results && <VisualizationSection />}
      </div>
    </div>
  );
}

export default App;
