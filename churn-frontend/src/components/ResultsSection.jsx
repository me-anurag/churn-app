import React from "react";

const ResultsSection = ({ results }) => {
  if (!results) return null;

  const {
    model_used,
    training_accuracy,
    testing_accuracy,
    roc_auc,
    mean_prediction_probability,
    confusion_matrix,
    classification_report,
    business_summary,
  } = results;

  return (
    <div
      style={{
        marginTop: "40px",
        padding: "30px",
        borderRadius: "15px",
        background: "linear-gradient(135deg, #d9a7c7, #fffcdc)",
        boxShadow: "0 6px 20px rgba(0,0,0,0.2)",
      }}
    >
      <h2
        style={{
          textAlign: "center",
          fontSize: "28px",
          marginBottom: "20px",
        }}
      >
        📊 Prediction Results
      </h2>

      {/* METRICS SECTION */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          padding: "20px",
        }}
      >
        {/* LEFT BLOCK */}
        <div style={{ width: "48%" }}>
          <h3>Model Information</h3>
          <p><b>Model Used:</b> {model_used}</p>
          <p><b>Training Accuracy:</b> {training_accuracy}</p>
          <p><b>Testing Accuracy:</b> {testing_accuracy}</p>
          <p><b>ROC-AUC Score:</b> {roc_auc}</p>
          <p><b>Mean Churn Probability:</b> {mean_prediction_probability}</p>

          <br />

          <h3>Business Interpretation</h3>
          <div
            style={{
              background: "#fff5f5",
              padding: "15px",
              borderRadius: "10px",
              whiteSpace: "pre-line",
              fontFamily: "monospace",
              boxShadow: "0 0 8px rgba(0,0,0,0.2)",
            }}
          >
            {business_summary}
          </div>
        </div>

        {/* RIGHT BLOCK */}
        <div style={{ width: "48%" }}>
          <h3>Confusion Matrix</h3>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(2, 80px)",
              gap: "10px",
              background: "white",
              padding: "20px",
              borderRadius: "10px",
              textAlign: "center",
              boxShadow: "0 0 8px rgba(0,0,0,0.2)",
            }}
          >
            <div style={{ fontWeight: "bold" }}>TN</div>
            <div style={{ fontWeight: "bold" }}>FP</div>
            <div style={{ fontWeight: "bold" }}>FN</div>
            <div style={{ fontWeight: "bold" }}>TP</div>

            {/* Values */}
            {confusion_matrix.flat().map((cell, idx) => (
              <div
                key={idx}
                style={{
                  padding: "12px",
                  background: "#e3e3e3",
                  borderRadius: "6px",
                }}
              >
                {cell}
              </div>
            ))}
          </div>

          <br />

          <h3>Classification Report</h3>
          <pre
            style={{
                background: "#f0fff0",
                padding: "15px",
                borderRadius: "10px",
                overflowY: "scroll",
                height: "220px",
                fontFamily: "Courier New, monospace",
                fontSize: "15px",
                whiteSpace: "pre",   // IMPORTANT
                boxShadow: "0 0 8px rgba(0,0,0,0.2)",
            }}
            >
            {classification_report}
            </pre>

        </div>
      </div>
    </div>
  );
};

export default ResultsSection;
