import React from "react";

const InfoSection = () => {
  const cards = [
    { title: "What is Churn?", text: "Churn prediction identifies customers likely to leave a service." },
    { title: "Dataset Used", text: "Demographics, tenure, charges, contract type, payment methods, etc." },
    { title: "Algorithms", text: "Logistic Regression, Random Forest, XGBoost, SVM, Neural Networks." },
    { title: "Benefits", text: "Reduce churn, increase retention, improve business decisions." },
    { title: "Use Cases", text: "Telecom, SaaS, OTT, Banking, Subscription services." }
  ];

  return (
    <div style={{ marginTop: "40px" }}>
      <h2 style={{
        fontSize: "30px",
        color: "white",
        marginBottom: "25px",
        textShadow: "0px 3px 6px rgba(0,0,0,0.3)"
      }}>
        Churn Prediction Overview
      </h2>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "25px",
        }}
      >
        {cards.map((c, i) => (
          <div
            key={i}
            className="info-card"
            style={{
              background: "linear-gradient(145deg, #ffffff, #e3e3e3)",
              padding: "25px",
              borderRadius: "12px",
              boxShadow: "0 6px 16px rgba(0,0,0,0.25)",
              transition: "0.3s",
              cursor: "pointer",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "scale(1.05)";
              e.currentTarget.style.boxShadow = "0 12px 24px rgba(0,0,0,0.3)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "scale(1)";
              e.currentTarget.style.boxShadow = "0 6px 16px rgba(0,0,0,0.25)";
            }}
          >
            <h3 style={{ marginBottom: "12px", fontSize: "20px" }}>{c.title}</h3>
            <p>{c.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default InfoSection;
