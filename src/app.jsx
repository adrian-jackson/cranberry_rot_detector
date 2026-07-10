import { useState, useRef, useCallback } from "react";

// Reads from environment variable set by Docker,
// falls back to localhost for local development

// const API = import.meta.env.VITE_API_URL || "https://adrian-jackson--cranberry-inspector-cranberryinspector-predict.modal.run"; 
const API = import.meta.env.VITE_API_URL || " https://adrian-jackson--cranberry-inspector-cranberryinspector-f-db90e3.modal.run";

// Colour matches the backend overlay
const CLASS_COLOR = { 0: "#ff5050", 1: "#50ff50" };
const CLASS_LABEL = { 0: "Rot", 1: "Ripe" };

export default function App() {
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);
  const [hovered,  setHovered]  = useState(null);  // cranberry under cursor
  const [tooltip,  setTooltip]  = useState({ x: 0, y: 0 });
  const imgRef = useRef(null);

  // ── Upload + predict ────────────────────────────────────────────────────
  const handleFile = useCallback(async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setHovered(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const res  = await fetch(`${API}/predict`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "HTTP request failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Mouse move over image: find nearest bounding box ────────────────────
  const handleMouseMove = useCallback((e) => {
    if (!result || !imgRef.current) return;

    const rect    = imgRef.current.getBoundingClientRect();
    const [imgW, imgH] = result.image_size;

    // Scale from displayed size to model output size
    const scaleX  = imgW / rect.width;
    const scaleY  = imgH / rect.height;
    const mx      = (e.clientX - rect.left)  * scaleX;
    const my      = (e.clientY - rect.top)   * scaleY;

    // Find the cranberry whose bbox contains the cursor
    const hit = result.cranberries.find(({ bbox }) => {
      const [x1, y1, x2, y2] = bbox;
      return mx >= x1 && mx <= x2 && my >= y1 && my <= y2;
    });

    setHovered(hit || null);
    setTooltip({ x: e.clientX, y: e.clientY });
  }, [result]);

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div style={styles.page}>
      <h1 style={styles.title}>🍒 Cranberry Quality Inspector</h1>

      {/* Upload */}
      <label style={styles.uploadBtn}>
        {loading ? "Analysing…" : "Upload Image"}
        <input type="file" accept="image/*"
               onChange={handleFile} style={{ display: "none" }} />
      </label>

      {error && <p style={styles.error}>{error}</p>}

      {/* Annotated image + hover */}
      {result && (
        <div style={styles.layout}>

          {/* Image panel */}
          <div style={{ position: "relative", display: "inline-block" }}>
            <img
              ref={imgRef}
              src={`data:image/png;base64,${result.annotated_image}`}
              alt="annotated"
              style={styles.img}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHovered(null)}
            />

            {/* Tooltip */}
            {hovered && (
              <div style={{
                ...styles.tooltip,
                left: tooltip.x + 12,
                top:  tooltip.y - 10,
              }}>
                <div style={{
                  fontWeight: "bold",
                  color: CLASS_COLOR[hovered.predicted_class],
                  marginBottom: 4,
                }}>
                  {CLASS_LABEL[hovered.predicted_class]}
                  {" "}({(hovered.confidence * 100).toFixed(1)}% confidence)
                </div>
                <ProbBar label="Rot"  value={hovered.p_rot}  color="#ff5050" />
                <ProbBar label="Ripe" value={hovered.p_ripe} color="#50ff50" />
                <div style={{ fontSize: 11, color: "#aaa", marginTop: 4 }}>
                  SAM score: {(hovered.sam_score * 100).toFixed(1)}%
                </div>
              </div>
            )}
          </div>

          {/* Summary panel */}
          <div style={styles.summary}>
            <h2 style={{ marginTop: 0 }}>Summary</h2>
            <Stat label="Total detected" value={result.summary.total} />
            <Stat label="Ripe"  value={result.summary.n_ripe} color="#50ff50" />
            <Stat label="Rot"   value={result.summary.n_rot}  color="#ff5050" />
            <Stat label="% Rot" value={`${result.summary.pct_rot}%`}
                  color={result.summary.pct_rot > 20 ? "#ff5050" : "#50ff50"} />

            <h3 style={{ marginTop: 24 }}>Per-Berry</h3>
            <div style={styles.berryList}>
              {result.cranberries.map((c) => (
                <div key={c.mask_index}
                     style={{
                       ...styles.berryRow,
                       borderLeft: `4px solid ${CLASS_COLOR[c.predicted_class]}`,
                       background: hovered?.mask_index === c.mask_index
                                   ? "#2a2a2a" : "transparent",
                     }}
                     onMouseEnter={() => setHovered(c)}
                     onMouseLeave={() => setHovered(null)}
                >
                  <span style={{ color: CLASS_COLOR[c.predicted_class] }}>
                    #{c.mask_index} {CLASS_LABEL[c.predicted_class]}
                  </span>
                  <span style={{ color: "#aaa", fontSize: 12 }}>
                    rot {(c.p_rot * 100).toFixed(0)}% /
                    ripe {(c.p_ripe * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Small reusable components ──────────────────────────────────────────────
function ProbBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between",
                    fontSize: 12, marginBottom: 2 }}>
        <span>{label}</span>
        <span>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ background: "#333", borderRadius: 4, height: 8 }}>
        <div style={{
          width: `${value * 100}%`, height: "100%",
          background: color, borderRadius: 4,
          transition: "width 0.2s",
        }} />
      </div>
    </div>
  );
}

function Stat({ label, value, color = "white" }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between",
                  padding: "6px 0", borderBottom: "1px solid #333" }}>
      <span style={{ color: "#aaa" }}>{label}</span>
      <span style={{ color, fontWeight: "bold" }}>{value}</span>
    </div>
  );
}

// ── Styles ─────────────────────────────────────────────────────────────────
const styles = {
  page: {
    minHeight: "100vh", background: "#111", color: "white",
    fontFamily: "system-ui, sans-serif", padding: "32px",
    boxSizing: "border-box",
  },
  title: { textAlign: "center", marginBottom: 24 },
  uploadBtn: {
    display: "block", margin: "0 auto 24px", padding: "12px 32px",
    background: "#e63946", color: "white", borderRadius: 8,
    cursor: "pointer", fontWeight: "bold", fontSize: 16,
    width: "fit-content", textAlign: "center",
  },
  error:  { color: "#ff5050", textAlign: "center" },
  layout: { display: "flex", gap: 24, alignItems: "flex-start",
             flexWrap: "wrap", justifyContent: "center" },
  img:    { maxWidth: "70vw", maxHeight: "80vh",
             borderRadius: 8, display: "block", cursor: "crosshair" },
  tooltip: {
    position: "fixed", zIndex: 1000,
    background: "#1e1e1e", border: "1px solid #444",
    borderRadius: 8, padding: "10px 14px",
    pointerEvents: "none", minWidth: 180,
    boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
  },
  summary: {
    minWidth: 220, maxWidth: 300, background: "#1a1a1a",
    borderRadius: 8, padding: 20, flexShrink: 0,
  },
  berryList: { maxHeight: 400, overflowY: "auto" },
  berryRow: {
    padding: "6px 10px", marginBottom: 4, borderRadius: 4,
    cursor: "default", display: "flex",
    justifyContent: "space-between", alignItems: "center",
    fontSize: 13,
  },
};