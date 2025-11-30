import React, { useState, useRef, useEffect } from "react";

const UploadSection = ({ setResults }) => {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState("");
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);

  const monitorRef = useRef(null);

  // Auto-scroll system monitor
  useEffect(() => {
    if (monitorRef.current) {
      monitorRef.current.scrollTop = monitorRef.current.scrollHeight;
    }
  }, [logs]);

  // Add one log line with delay
  const addLog = (text, delay = 800) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        setLogs((prev) => [...prev, text]);
        resolve();
      }, delay);
    });
  };

  // Log animation sequence
  const runAnimationLogs = async () => {
    setLogs([]);
    await addLog("🔍 Loading dataset...");
    await addLog("📊 Inspecting dataset...");
    await addLog("🧽 Cleaning missing values...");
    await addLog("🔧 Encoding categorical variables...");
    await addLog("📏 Scaling numeric features...");
    await addLog("✂ Splitting into train/test sets...");
    await addLog("🤖 Training model...");
    await addLog("📈 Calculating ROC-AUC...");
    await addLog("🔎 Evaluating model...");
    await addLog("✔ Processing complete. Fetching results...", 1200);
  };

  // Handle backend request
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file || !model) {
      alert("Please upload CSV and select a model.");
      return;
    }

    setLoading(true);
    await runAnimationLogs();

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", model);
    formData.append("target_column", "Churn");

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        alert("Backend Error: " + err.detail);
        setLoading(false);
        return;
      }

      const data = await res.json();
      console.log("Prediction Response:", data);

      setResults(data);
    } catch (error) {
      alert("Backend error: " + error);
      console.error(error);
    }

    setLoading(false);
  };

  return (
    <div
      style={{
        background: "linear-gradient(135deg, #ff9a9e, #fad0c4)",
        padding: "30px",
        borderRadius: "15px",
        marginTop: "40px",
        boxShadow: "0 6px 20px rgba(0,0,0,0.3)",
        display: "flex",
        justifyContent: "space-between",
      }}
    >
      {/* LEFT SIDE (UPLOAD + MODEL SELECT) */}
      <div style={{ width: "48%" }}>
        <h2 style={{ marginBottom: "15px" }}>Upload Dataset & Choose Model</h2>

        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
          style={{
            padding: "10px",
            background: "white",
            borderRadius: "8px",
            marginBottom: "15px",
          }}
        />

        <br />

        <select
          onChange={(e) => setModel(e.target.value)}
          style={{
            padding: "10px",
            width: "230px",
            borderRadius: "8px",
            marginTop: "10px",
            marginBottom: "20px",
          }}
        >
          <option value="">Select Model</option>
          <option value="logistic_regression">Logistic Regression</option>
          <option value="naive_bayes">Naive Bayes</option>
          <option value="knn">KNN</option>
          <option value="svm">SVM</option>
          <option value="decision_tree">Decision Tree</option>
          <option value="random_forest">Random Forest</option>
          <option value="xgboost">XGBoost</option>
          <option value="lightgbm">LightGBM</option>
        </select>

        <br />

        <button className="predict-btn" onClick={handleSubmit}>
          {loading ? "Processing..." : "Begin Churn Prediction"}
        </button>
      </div>

      {/* RIGHT SIDE — MINI MONITOR */}
      <div
        ref={monitorRef}
        style={{
          width: "48%",
          background: "black",
          color: "#00ff00",
          fontFamily: "monospace",
          padding: "15px",
          borderRadius: "10px",
          height: "260px",
          overflowY: "auto",
          boxShadow: "0 0 10px #00ff00aa inset",
        }}
      >
        <h3 style={{ color: "#00ff00" }}>System Monitor</h3>
        <hr style={{ borderColor: "#00ff00" }} />

        {logs.map((log, idx) => (
          <p key={idx}>{log}</p>
        ))}
      </div>
    </div>
  );
};

export default UploadSection;
