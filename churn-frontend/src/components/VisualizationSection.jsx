import React, { useState, useEffect } from "react";

const VisualizationSection = () => {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [title, setTitle] = useState("");
  const [graphMessage, setGraphMessage] = useState("");
  const [customerCount, setCustomerCount] = useState(0);
  const [selectedCustomer, setSelectedCustomer] = useState(0);
  const [textResult, setTextResult] = useState("");

  // Fetch how many customers are in X_test (for dropdown)
  useEffect(() => {
    const fetchCount = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/explain/customer_count");
        const data = await res.json();
        setCustomerCount(data.count || 0);
      } catch (err) {
        console.error("Error fetching customer count:", err);
      }
    };
    fetchCount();
  }, []);

  // For image-based plots (normal graphs + SHAP plots)
  const fetchImage = async (endpoint, graphTitle) => {
    setLoading(true);
    setTitle(graphTitle);
    setTextResult("");
    setGraphMessage("");
    setImage(null);

    try {
      const isShap = endpoint.startsWith("shap");
      const baseUrl = isShap
        ? "http://127.0.0.1:8000/explain/"
        : "http://127.0.0.1:8000/visualize/";

      const res = await fetch(baseUrl + endpoint);
      const data = await res.json();

      if (data.image === null && data.message) {
        setGraphMessage(data.message);
        setImage(null);
      } else if (data.image) {
        setGraphMessage("");
        setImage(data.image);
      } else {
        setGraphMessage("No visualization data returned.");
      }
    } catch (err) {
      console.error("Error fetching image:", err);
      setGraphMessage("Failed to load visualization.");
    }

    setLoading(false);
  };

  // For text-only SHAP explanation
  const fetchText = async (endpoint) => {
    setLoading(true);
    setTitle("SHAP Text Explanation");
    setGraphMessage("");
    setImage(null);
    setTextResult("");

    try {
      const res = await fetch("http://127.0.0.1:8000/explain/" + endpoint);
      const data = await res.json();
      setTextResult(data.text || "No explanation available.");
    } catch (err) {
      console.error("Error fetching SHAP text:", err);
      setTextResult("Error loading explanation.");
    }

    setLoading(false);
  };

  return (
    <div
      style={{
        marginTop: "40px",
        padding: "30px",
        background: "linear-gradient(140deg, #a8edea, #fed6e3)",
        borderRadius: "18px",
        boxShadow: "0 5px 20px rgba(0,0,0,0.25)",
      }}
    >
      <h2
        style={{
          textAlign: "center",
          fontSize: "28px",
          marginBottom: "20px",
          fontWeight: "600",
        }}
      >
        📊 Visualization & Explainability Dashboard
      </h2>

      {/* CUSTOMER DROPDOWN */}
      <div style={{ marginBottom: "20px", textAlign: "center" }}>
        <label style={{ fontSize: "18px", marginRight: "10px" }}>
          Select Customer:
        </label>
        <select
          value={selectedCustomer}
          onChange={(e) => setSelectedCustomer(parseInt(e.target.value))}
          style={{
            padding: "8px",
            fontSize: "16px",
            borderRadius: "8px",
            border: "1px solid #999",
          }}
        >
          {customerCount > 0 ? (
            [...Array(customerCount).keys()].map((i) => (
              <option key={i} value={i}>
                Customer #{i}
              </option>
            ))
          ) : (
            <option value={0}>Run prediction first</option>
          )}
        </select>
      </div>

      {/* BUTTONS */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-around",
          flexWrap: "wrap",
          gap: "15px",
          marginBottom: "20px",
        }}
      >
        {/* Standard visualizations */}
        <button
          className="viz-btn"
          onClick={() => fetchImage("roc", "ROC Curve")}
        >
          ROC Curve
        </button>

        <button
          className="viz-btn"
          onClick={() =>
            fetchImage("confusion_heatmap", "Confusion Matrix Heatmap")
          }
        >
          Confusion Heatmap
        </button>

        <button
          className="viz-btn"
          onClick={() =>
            fetchImage("correlation_heatmap", "Correlation Heatmap")
          }
        >
          Correlation Heatmap
        </button>

        <button
          className="viz-btn"
          onClick={() =>
            fetchImage("churn_distribution", "Churn Distribution")
          }
        >
          Churn Distribution
        </button>

        <button
          className="viz-btn"
          onClick={() =>
            fetchImage("feature_importance", "Feature Importance")
          }
        >
          Feature Importance
        </button>

        {/* SHAP visualizations */}
        <button
          className="viz-btn"
          onClick={() =>
            fetchImage(
              `shap_waterfall?index=${selectedCustomer}`,
              "SHAP Waterfall Plot"
            )
          }
        >
          SHAP Waterfall
        </button>

        <button
          className="viz-btn"
          onClick={() =>
            fetchImage(
              `shap_decision?index=${selectedCustomer}`,
              "SHAP Decision Plot"
            )
          }
        >
          SHAP Decision Plot
        </button>

        {/* SHAP text explanation */}
        <button
          className="viz-btn"
          onClick={() =>
            fetchText(`shap_text?index=${selectedCustomer}`)
          }
        >
          SHAP Explanation (Text)
        </button>
      </div>

      {/* DISPLAY AREA */}
      <div style={{ textAlign: "center", marginTop: "25px" }}>
        {loading && (
          <p
            style={{
              fontSize: "18px",
              fontFamily: "monospace",
              color: "#444",
            }}
          >
            Loading Visualization...
          </p>
        )}

        {title && !loading && (image || graphMessage || textResult) && (
          <h3 style={{ marginBottom: "15px" }}>{title}</h3>
        )}

        {/* Graph-related messages (e.g., feature importance not available) */}
        {graphMessage && !loading && (
          <div
            style={{
              marginTop: "10px",
              padding: "15px",
              background: "#ffffff",
              borderRadius: "12px",
              boxShadow: "0 3px 10px rgba(0,0,0,0.15)",
              fontSize: "18px",
              color: "#333",
              maxWidth: "650px",
              marginLeft: "auto",
              marginRight: "auto",
              whiteSpace: "pre-line",
            }}
          >
            {graphMessage}
          </div>
        )}

        {/* Text explanation block (SHAP explanation card) */}
        {textResult && !loading && (
          <pre
            style={{
              marginTop: "20px",
              background: "#fff",
              padding: "15px",
              borderRadius: "12px",
              fontSize: "16px",
              boxShadow: "0 2px 10px rgba(0,0,0,0.15)",
              maxWidth: "700px",
              marginLeft: "auto",
              marginRight: "auto",
              textAlign: "left",
              whiteSpace: "pre-wrap",
            }}
          >
            {textResult}
          </pre>
        )}

        {/* Image display */}
        {!loading && image && (
          <div>
            <img
              src={`data:image/png;base64,${image}`}
              alt={title}
              style={{
                maxWidth: "95%",
                borderRadius: "12px",
                boxShadow: "0 0 15px rgba(0,0,0,0.3)",
                marginTop: "20px",
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default VisualizationSection;
