// frontend/src/App.jsx
// Replace the entire contents of this file with this code.

import { useState } from "react"

const API = "http://127.0.0.1:8000"

const REGIONS = [
  "North India",
  "South India",
  "East India",
  "West India",
  "Central India",
]

// ── Small reusable components ────────────────────────────────────

function Slider({ label, name, min, max, step, value, onChange, unit }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontWeight: 600, fontSize: 13, color: "#2E7D32" }}>{label}</span>
        <span style={{
          fontWeight: 700, fontSize: 13, color: "#fff",
          background: "#43A047", borderRadius: 10, padding: "1px 9px"
        }}>
          {value}{unit}
        </span>
      </div>
      <input
        type="range" name={name} min={min} max={max} step={step}
        value={value} onChange={onChange}
        style={{ width: "100%", accentColor: "#43A047", cursor: "pointer" }}
      />
    </div>
  )
}

function Badge({ text, color }) {
  const palette = {
    green: { bg: "#E8F5E9", text: "#1B5E20", border: "#A5D6A7" },
    orange: { bg: "#FFF3E0", text: "#E65100", border: "#FFCC80" },
    red: { bg: "#FFEBEE", text: "#B71C1C", border: "#EF9A9A" },
    blue: { bg: "#E3F2FD", text: "#0D47A1", border: "#90CAF9" },
  }
  const c = palette[color] || palette.green
  return (
    <span style={{
      background: c.bg, color: c.text, border: `1px solid ${c.border}`,
      padding: "3px 12px", borderRadius: 20, fontWeight: 700,
      fontSize: 12, display: "inline-block"
    }}>
      {text}
    </span>
  )
}

function Card({ children, style }) {
  return (
    <div style={{
      background: "#fff", borderRadius: 14, padding: 20,
      boxShadow: "0 2px 12px rgba(0,0,0,0.08)",
      ...style
    }}>
      {children}
    </div>
  )
}

function SectionTitle({ emoji, text }) {
  return (
    <h3 style={{
      color: "#1B5E20", margin: "0 0 12px",
      fontSize: 15, fontWeight: 700,
      display: "flex", alignItems: "center", gap: 6
    }}>
      <span>{emoji}</span> {text}
    </h3>
  )
}

// ── Main App ────────────────────────────────────────────────────

export default function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const [fields, setFields] = useState({
    temperature: 30,
    humidity: 70,
    rainfall: 50,
    soil_ph: 6.5,
    nitrogen: 50,
    phosphorus: 30,
    region: 0,
    days_sowing: 60,
  })

  const handleImg = (e) => {
    const f = e.target.files[0]
    if (!f) return
    setImage(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError("")
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (!f) return
    setImage(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError("")
  }

  const handleField = (e) => {
    const { name, value } = e.target
    setFields((p) => ({ ...p, [name]: parseFloat(value) }))
  }

  const handlePredict = async () => {
    if (!image) { setError("Please upload a leaf image first."); return }
    setLoading(true)
    setError("")
    setResult(null)

    const fd = new FormData()
    fd.append("file", image)
    Object.entries(fields).forEach(([k, v]) => fd.append(k, v))

    try {
      const res = await fetch(`${API}/predict`, { method: "POST", body: fd })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || "API returned an error")
      setResult(data)
    } catch (err) {
      setError("Error: " + err.message + ". Make sure the FastAPI server is running.")
    } finally {
      setLoading(false)
    }
  }

  const severityColor = (s) =>
    s === "severe" ? "red" : s === "moderate" ? "orange" : "green"

  const formatDisease = (d) =>
    d ? d.replace(/_{2,}/g, " → ").replace(/_/g, " ") : ""

  return (
    <div style={{
      fontFamily: "'Segoe UI', Arial, sans-serif",
      minHeight: "100vh",
      background: "linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%)",
      padding: "24px 16px",
    }}>

      {/* ── Header ── */}
      <div style={{ textAlign: "center", marginBottom: 28 }}>
        <div style={{ fontSize: 44, marginBottom: 6 }}>🌿</div>
        <h1 style={{
          margin: 0, fontSize: 28, fontWeight: 800,
          color: "#1B5E20", letterSpacing: -0.5
        }}>
          Crop Disease Detector
        </h1>
        <p style={{ margin: "6px 0 0", color: "#555", fontStyle: "italic", fontSize: 14 }}>
          फसल रोग पहचान एवं उत्पादन हानि अनुमान प्रणाली
        </p>
        <p style={{ margin: "4px 0 0", color: "#888", fontSize: 12 }}>
          CNN + XGBoost Hybrid Pipeline · ResNet-50 · PlantVillage Dataset
        </p>
      </div>

      {/* ── Main Grid ── */}
      <div style={{
        maxWidth: 1000, margin: "0 auto",
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 20,
      }}>

        {/* ── LEFT PANEL ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Upload Box */}
          <Card>
            <SectionTitle emoji="📷" text="Upload Leaf Photo" />
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => document.getElementById("fileInput").click()}
              style={{
                border: "2px dashed #81C784",
                borderRadius: 10,
                padding: preview ? 8 : 30,
                textAlign: "center",
                cursor: "pointer",
                background: "#F9FBE7",
                transition: "background 0.2s",
                minHeight: 160,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <input
                id="fileInput" type="file" accept="image/*"
                style={{ display: "none" }} onChange={handleImg}
              />
              {preview ? (
                <img
                  src={preview} alt="Leaf preview"
                  style={{
                    maxHeight: 220, maxWidth: "100%",
                    borderRadius: 8, objectFit: "contain"
                  }}
                />
              ) : (
                <div>
                  <div style={{ fontSize: 36, marginBottom: 8 }}>🍃</div>
                  <div style={{ color: "#66BB6A", fontWeight: 600, fontSize: 14 }}>
                    Click or drag & drop a leaf image
                  </div>
                  <div style={{ color: "#AAA", fontSize: 12, marginTop: 4 }}>
                    JPEG or PNG · Any size
                  </div>
                </div>
              )}
            </div>
            {preview && (
              <button
                onClick={() => document.getElementById("fileInput").click()}
                style={{
                  marginTop: 8, width: "100%", padding: "6px 0",
                  background: "none", border: "1px solid #A5D6A7",
                  borderRadius: 6, color: "#43A047", cursor: "pointer",
                  fontSize: 12, fontWeight: 600
                }}
              >
                Change Image
              </button>
            )}
          </Card>

          {/* Field Conditions */}
          <Card>
            <SectionTitle emoji="🌾" text="Field Conditions" />

            <Slider label="Temperature" name="temperature"
              min={15} max={45} step={0.5} unit="°C"
              value={fields.temperature} onChange={handleField} />

            <Slider label="Humidity" name="humidity"
              min={20} max={100} step={1} unit="%"
              value={fields.humidity} onChange={handleField} />

            <Slider label="Rainfall" name="rainfall"
              min={0} max={300} step={5} unit=" mm"
              value={fields.rainfall} onChange={handleField} />

            <Slider label="Soil pH" name="soil_ph"
              min={4} max={9} step={0.1} unit=""
              value={fields.soil_ph} onChange={handleField} />

            <Slider label="Nitrogen" name="nitrogen"
              min={0} max={120} step={1} unit=" kg/ha"
              value={fields.nitrogen} onChange={handleField} />

            <Slider label="Phosphorus" name="phosphorus"
              min={0} max={80} step={1} unit=" kg/ha"
              value={fields.phosphorus} onChange={handleField} />

            <Slider label="Days Since Sowing" name="days_sowing"
              min={10} max={150} step={1} unit=" days"
              value={fields.days_sowing} onChange={handleField} />

            <div style={{ marginTop: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: "#2E7D32", marginBottom: 6 }}>
                Region
              </div>
              <select
                name="region" value={fields.region} onChange={handleField}
                style={{
                  width: "100%", padding: "8px 10px", borderRadius: 8,
                  border: "1px solid #A5D6A7", fontSize: 13,
                  background: "#F9FBE7", color: "#333", cursor: "pointer"
                }}
              >
                {REGIONS.map((r, i) => (
                  <option key={i} value={i}>{r}</option>
                ))}
              </select>
            </div>
          </Card>

          {/* Detect Button */}
          <button
            onClick={handlePredict}
            disabled={loading || !image}
            style={{
              width: "100%", padding: "15px 0",
              background: loading || !image
                ? "#A5D6A7"
                : "linear-gradient(135deg, #2E7D32, #43A047)",
              color: "#fff", border: "none", borderRadius: 10,
              fontSize: 16, fontWeight: 700, cursor: loading || !image ? "not-allowed" : "pointer",
              boxShadow: loading || !image ? "none" : "0 4px 12px rgba(46,125,50,0.35)",
              transition: "all 0.2s",
              letterSpacing: 0.3,
            }}
          >
            {loading ? "⏳  Analysing leaf..." : "🔍  Detect Disease & Predict Yield"}
          </button>

          {error && (
            <div style={{
              background: "#FFEBEE", border: "1px solid #EF9A9A",
              borderRadius: 8, padding: "10px 14px",
              color: "#B71C1C", fontSize: 13
            }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {result ? (
            <>
              {/* Disease Result */}
              <Card style={{ borderTop: "4px solid #2E7D32" }}>
                <SectionTitle emoji="🦠" text="Disease Detection" />
                <div style={{
                  fontSize: 17, fontWeight: 700, color: "#1B5E20",
                  marginBottom: 10, lineHeight: 1.3
                }}>
                  {formatDisease(result.disease)}
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <Badge
                    text={`${(result.confidence * 100).toFixed(1)}% confidence`}
                    color="blue"
                  />
                  <Badge
                    text={result.severity.toUpperCase()}
                    color={severityColor(result.severity)}
                  />
                  <Badge
                    text={`Yield loss: ${result.yield_loss_pct}%`}
                    color={result.yield_loss_pct > 40 ? "red" : result.yield_loss_pct > 20 ? "orange" : "green"}
                  />
                </div>

                {/* Yield loss bar */}
                <div style={{ marginTop: 14 }}>
                  <div style={{
                    display: "flex", justifyContent: "space-between",
                    fontSize: 12, color: "#666", marginBottom: 4
                  }}>
                    <span>Estimated Yield Loss</span>
                    <span style={{ fontWeight: 700 }}>{result.yield_loss_pct}%</span>
                  </div>
                  <div style={{
                    height: 10, background: "#E8F5E9",
                    borderRadius: 5, overflow: "hidden"
                  }}>
                    <div style={{
                      height: "100%", borderRadius: 5,
                      width: `${Math.min(result.yield_loss_pct, 100)}%`,
                      background: result.yield_loss_pct > 40
                        ? "linear-gradient(90deg,#EF5350,#B71C1C)"
                        : result.yield_loss_pct > 20
                          ? "linear-gradient(90deg,#FFA726,#E65100)"
                          : "linear-gradient(90deg,#66BB6A,#2E7D32)",
                      transition: "width 0.6s ease"
                    }} />
                  </div>
                </div>
              </Card>

              {/* Recommendation */}
              <Card>
                <SectionTitle emoji="💊" text="Treatment Recommendation" />
                {[
                  ["🦠 Pathogen", result.recommendation.pathogen],
                  ["🧪 Pesticide", result.recommendation.pesticide],
                  ["🗓️ Frequency", result.recommendation.frequency],
                  ["🌱 Fertilizer", result.recommendation.fertilizer],
                  ["⚡ Action", result.recommendation.action],
                  ["📞 Helpline", result.recommendation.helpline],
                ].map(([label, value]) => value && value !== "N/A" && (
                  <div key={label} style={{
                    marginBottom: 10, padding: "8px 10px",
                    background: "#F9FBE7", borderRadius: 8,
                    borderLeft: "3px solid #81C784"
                  }}>
                    <div style={{ fontWeight: 700, fontSize: 11, color: "#555", marginBottom: 2 }}>
                      {label}
                    </div>
                    <div style={{ fontSize: 13, color: "#333", lineHeight: 1.4 }}>
                      {value}
                    </div>
                  </div>
                ))}

                {result.recommendation.hindi_summary && (
                  <div style={{
                    marginTop: 8, padding: "10px 12px",
                    background: "#FFF8E1", borderRadius: 8,
                    border: "1px solid #FFE082",
                    fontSize: 13, color: "#555", fontStyle: "italic"
                  }}>
                    🇮🇳 {result.recommendation.hindi_summary}
                  </div>
                )}
              </Card>

              {/* Top-5 Predictions */}
              <Card>
                <SectionTitle emoji="📊" text="Top-5 Predictions" />
                {result.top5.map((item, i) => (
                  <div key={i} style={{ marginBottom: 10 }}>
                    <div style={{
                      display: "flex", justifyContent: "space-between",
                      fontSize: 12, marginBottom: 3
                    }}>
                      <span style={{
                        color: i === 0 ? "#1B5E20" : "#666",
                        fontWeight: i === 0 ? 700 : 400,
                        maxWidth: "78%", lineHeight: 1.3
                      }}>
                        {i === 0 && "✓ "}
                        {formatDisease(item.disease_class)}
                      </span>
                      <span style={{
                        fontWeight: 700,
                        color: i === 0 ? "#1B5E20" : "#888"
                      }}>
                        {(item.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{
                      height: 7, background: "#F1F8E9",
                      borderRadius: 4, overflow: "hidden"
                    }}>
                      <div style={{
                        height: "100%", borderRadius: 4,
                        width: `${Math.max(item.probability * 100, item.probability > 0 ? 1 : 0)}%`,
                        background: i === 0
                          ? "linear-gradient(90deg,#43A047,#1B5E20)"
                          : "#A5D6A7",
                        transition: "width 0.5s ease"
                      }} />
                    </div>
                  </div>
                ))}
              </Card>
            </>
          ) : (
            /* Placeholder when no result */
            <Card style={{
              display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              minHeight: 400, textAlign: "center",
              border: "2px dashed #C8E6C9"
            }}>
              <div style={{ fontSize: 56, marginBottom: 12 }}>🌱</div>
              <div style={{ fontWeight: 700, color: "#43A047", fontSize: 16, marginBottom: 8 }}>
                Results will appear here
              </div>
              <div style={{ color: "#AAA", fontSize: 13, lineHeight: 1.6, maxWidth: 240 }}>
                Upload a leaf photo, adjust the field conditions on the left, then click
                <strong style={{ color: "#43A047" }}> Detect Disease</strong>
              </div>
              <div style={{
                marginTop: 20, padding: "12px 16px",
                background: "#F1F8E9", borderRadius: 10,
                fontSize: 12, color: "#666", maxWidth: 260
              }}>
                <div style={{ fontWeight: 700, marginBottom: 4 }}>Supports detection of:</div>
                Tomato · Potato · Apple · Corn · Grape diseases
                and more across 38 disease classes
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* Footer */}
      <div style={{ textAlign: "center", marginTop: 28, color: "#AAA", fontSize: 12 }}>
        Built with ResNet-50 · XGBoost · Grad-CAM · FastAPI · React &nbsp;|&nbsp;
        PlantVillage Dataset · 38 Disease Classes
      </div>
    </div>
  )
}