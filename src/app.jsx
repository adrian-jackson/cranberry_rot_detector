import { useState, useRef, useCallback } from "react";
import { getStoredApiKey, setStoredApiKey, clearStoredApiKey } from "./apiKey";
import { exportResultsZip } from "./export";

const API = import.meta.env.VITE_API_URL || "https://adrian-jackson--cranberry-inspector-cranberryinspector-f-db90e3.modal.run";

const CLASS_COLOR = { 0: "#ff5050", 1: "#50ff50" };
const CLASS_LABEL = { 0: "Rot", 1: "Ripe" };

const BATCH_CONCURRENCY = 3;

// ── Access key gate ─────────────────────────────────────────────────────────
function ApiKeyGate({ onSubmit, invalid }) {
  const [value, setValue] = useState("");
  return (
    <div style={styles.page}>
      <div style={styles.keyGateBox}>
        <h2 style={{ marginTop: 0 }}>Access key required</h2>
        <p style={styles.modalText}>
          This tool is limited to people who've been given a key. Enter yours below —
          it's saved only in this browser, never sent anywhere except to the API.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (value.trim()) onSubmit(value.trim());
          }}
        >
          <input
            type="password"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Access key"
            style={styles.keyInput}
            autoFocus
          />
          <button type="submit" style={{ ...styles.uploadBtn, width: "100%", marginTop: 12, textAlign: "center" }}>
            Continue
          </button>
        </form>
        {invalid && (
          <p style={{ color: "#ff8080", fontSize: 13, marginTop: 12 }}>
            That key was rejected by the server. Double-check it and try again.
          </p>
        )}
      </div>
    </div>
  );
}

// ── How-to modal content ───────────────────────────────────────────────────
function HowToModal({ onClose }) {
  return (
    <div style={styles.modalOverlay} onClick={onClose}>
      <div style={styles.modalBox} onClick={e => e.stopPropagation()}>
        <h2 style={{ marginTop: 0, color: "white" }}>How to use</h2>

        <p style={styles.modalText}>
          <strong style={{ color: "#e63946" }}>1. Upload a photo</strong> of
          cranberries taken from directly above, in good lighting. The clearer
          the image, the more accurate the results. Or switch to{" "}
          <strong style={{ color: "#e63946" }}>Folder</strong> mode to process a
          whole folder of images at once.
        </p>
        <p style={styles.modalText}>
          <strong style={{ color: "#e63946" }}>2. Wait for analysis.</strong> The
          first request takes ~30 seconds while the model starts up. Subsequent
          uploads are faster.
        </p>
        <p style={styles.modalText}>
          <strong style={{ color: "#e63946" }}>3. Read the results.</strong> Each
          berry is outlined in <span style={{ color: "#50ff50" }}>green (ripe)</span> or{" "}
          <span style={{ color: "#ff5050" }}>red (rot)</span>. Hover over a berry
          in the image or the list to see its exact probability breakdown.
        </p>
        <p style={styles.modalText}>
          <strong style={{ color: "#e63946" }}>4. Toggle views.</strong> Use the
          Original / Annotated buttons to switch between the raw photo and the
          annotated overlay.
        </p>
        <p style={styles.modalText}>
          <strong style={{ color: "#e63946" }}>5. Export.</strong> Use the Export
          button to download a .zip with every annotated image plus a
          results.xlsx spreadsheet summarizing rot %, ripe %, average DINO
          confidence, and any detected label/barcode per image.
        </p>

        <hr style={{ borderColor: "#333", margin: "16px 0" }} />

        <p style={styles.modalText}>
          <strong style={{ color: "#aaa" }}>What is SAM score?</strong>
          <br />
          SAM (Segment Anything Model) detects and outlines each individual
          cranberry in the photo. The SAM score is its confidence that a detected
          region is actually a cranberry — higher is better. Berries with a low
          SAM score may be partially obscured, oddly shaped, or not cranberries
          at all.
        </p>

        <button style={styles.closeBtn} onClick={onClose}>Got it</button>
      </div>
    </div>
  );
}

export default function App() {
  const [apiKey,        setApiKey]        = useState(() => getStoredApiKey());
  const [keyInvalid,    setKeyInvalid]    = useState(false);

  const [mode,           setMode]          = useState("single"); // "single" | "batch"

  const [result,        setResult]        = useState(null);
  const [fileName,      setFileName]      = useState(null);
  const [originalUrl,   setOriginalUrl]   = useState(null);  // object URL of uploaded file
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState(null);
  const [hovered,       setHovered]       = useState(null);  // mask_index of hovered berry
  const [tooltip,       setTooltip]       = useState({ x: 0, y: 0 });
  const [showAnnotated, setShowAnnotated] = useState(true);  // toggle original/annotated
  const [showHowTo,     setShowHowTo]     = useState(false);
  const imgRef = useRef(null);

  const [batchResults,  setBatchResults]  = useState([]);   // [{filename, data, error}]
  const [batchProgress, setBatchProgress] = useState({ done: 0, total: 0 });
  const [batchLoading,  setBatchLoading]  = useState(false);

  // ── Auth helper ─────────────────────────────────────────────────────────
  const authedFetch = useCallback((path, opts = {}) => {
    return fetch(`${API}${path}`, {
      ...opts,
      headers: { ...(opts.headers || {}), "X-API-Key": apiKey },
    });
  }, [apiKey]);

  const handleAuthFailure = useCallback(() => {
    clearStoredApiKey();
    setApiKey(null);
    setKeyInvalid(true);
  }, []);

  // ── Upload + predict (single image) ─────────────────────────────────────
  const handleFile = useCallback(async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Show original image immediately so there's something to display
    // even while the API is running
    const objectUrl = URL.createObjectURL(file);
    setOriginalUrl(objectUrl);
    setFileName(file.name);
    setShowAnnotated(false);   // start on original while loading
    setLoading(true);
    setError(null);
    setResult(null);
    setHovered(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const res  = await authedFetch("/predict", { method: "POST", body: form });
      if (res.status === 401) { handleAuthFailure(); return; }
      const data = await res.json();
      console.log("API response:", data);

      if (!res.ok) throw new Error(data.detail || data.error || "Request failed");

      // Backend sets error:1 when no cranberries are detected
      if (data.error === 1) {
        setError("No cranberries detected, even after retrying with alternate prompts. Try a clearer photo taken from directly above.");
        setLoading(false);
        return;   // keep originalUrl displayed, don't set result
      }

      setResult(data);
      setShowAnnotated(true);  // switch to annotated view once results arrive
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [authedFetch, handleAuthFailure]);

  // ── Upload + predict (folder batch) ─────────────────────────────────────
  const handleFolder = useCallback(async (e) => {
    const files = Array.from(e.target.files).filter(f => f.type.startsWith("image/"));
    if (files.length === 0) return;

    setError(null);
    setBatchResults([]);
    setBatchProgress({ done: 0, total: files.length });
    setBatchLoading(true);

    const results = new Array(files.length);
    let nextIndex = 0;
    let authFailed = false;

    async function worker() {
      while (nextIndex < files.length && !authFailed) {
        const idx  = nextIndex++;
        const file = files[idx];
        try {
          const form = new FormData();
          form.append("file", file);
          const res = await authedFetch("/predict", { method: "POST", body: form });
          if (res.status === 401) { authFailed = true; return; }
          const data = await res.json();
          if (!res.ok) throw new Error(data.detail || "Request failed");
          results[idx] = {
            filename: file.name,
            data: data.error === 1 ? null : data,
            error: data.error === 1 ? "No cranberries detected" : null,
          };
        } catch (err) {
          results[idx] = { filename: file.name, data: null, error: err.message };
        }
        setBatchProgress(p => ({ ...p, done: p.done + 1 }));
      }
    }

    const workers = Array.from({ length: Math.min(BATCH_CONCURRENCY, files.length) }, worker);
    await Promise.all(workers);

    if (authFailed) {
      handleAuthFailure();
      setBatchLoading(false);
      return;
    }

    setBatchResults(results);
    setBatchLoading(false);
  }, [authedFetch, handleAuthFailure]);

  // ── Mouse move over image: find berry under cursor ─────────────────────
  const handleMouseMove = useCallback((e) => {
    if (!result || !imgRef.current || !showAnnotated) return;

    const rect         = imgRef.current.getBoundingClientRect();
    const [imgW, imgH] = result.image_size;
    const scaleX       = imgW / rect.width;
    const scaleY       = imgH / rect.height;
    const mx           = (e.clientX - rect.left) * scaleX;
    const my           = (e.clientY - rect.top)  * scaleY;

    const hit = result.cranberries.find(({ bounding_box }) => {
      const [x1, y1, x2, y2] = bounding_box;
      return mx >= x1 && mx <= x2 && my >= y1 && my <= y2;
    });

    setHovered(hit ? hit.mask_index : null);
    setTooltip({ x: e.clientX, y: e.clientY });
  }, [result, showAnnotated]);

  const hoveredBerry = result?.cranberries.find(c => c.mask_index === hovered) ?? null;

  const exportSingle = useCallback(() => {
    if (!result || !fileName) return;
    exportResultsZip([{ filename: fileName, data: result, error: null }]);
  }, [result, fileName]);

  const exportBatch = useCallback(() => {
    if (batchResults.length === 0) return;
    exportResultsZip(batchResults);
  }, [batchResults]);

  // ── Gate everything behind an access key ────────────────────────────────
  if (!apiKey) {
    return (
      <ApiKeyGate
        invalid={keyInvalid}
        onSubmit={(key) => {
          setStoredApiKey(key);
          setApiKey(key);
          setKeyInvalid(false);
        }}
      />
    );
  }

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div style={styles.page}>
      <h1 style={styles.title}>Cranberry Rot Detector</h1>

      {/* Mode toggle */}
      <div style={styles.toggleRow}>
        <button
          style={{ ...styles.toggleBtn, ...(mode === "single" ? styles.toggleActive : {}) }}
          onClick={() => setMode("single")}
        >
          Single Image
        </button>
        <button
          style={{ ...styles.toggleBtn, ...(mode === "batch" ? styles.toggleActive : {}) }}
          onClick={() => setMode("batch")}
        >
          Folder
        </button>
      </div>

      {/* Controls row */}
      <div style={styles.controls}>
        {mode === "single" ? (
          <label style={styles.uploadBtn}>
            {loading ? "Analysing…" : "Upload Image"}
            <input type="file" accept="image/*"
                   onChange={handleFile} style={{ display: "none" }}
                   disabled={loading} />
          </label>
        ) : (
          <label style={styles.uploadBtn}>
            {batchLoading ? "Analysing…" : "Choose Folder"}
            <input type="file" accept="image/*" multiple
                   webkitdirectory="" directory=""
                   onChange={handleFolder} style={{ display: "none" }}
                   disabled={batchLoading} />
          </label>
        )}

        {mode === "single" && result && (
          <button style={styles.howToBtn} onClick={exportSingle}>
            Export
          </button>
        )}
        {mode === "batch" && batchResults.length > 0 && !batchLoading && (
          <button style={styles.howToBtn} onClick={exportBatch}>
            Export ({batchResults.length})
          </button>
        )}

        <button style={styles.howToBtn} onClick={() => setShowHowTo(true)}>
          ? How to use
        </button>
      </div>

      {error && (
        <div style={styles.errorBox}>
          <p style={{ margin: 0 }}>{error}</p>
        </div>
      )}

      {/* ── Batch mode ─────────────────────────────────────────────────── */}
      {mode === "batch" && (batchLoading || batchResults.length > 0) && (
        <div style={styles.batchBox}>
          {batchLoading && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ marginBottom: 6, fontSize: 13, color: "#aaa" }}>
                Processing {batchProgress.done} / {batchProgress.total}…
              </div>
              <div style={styles.progressTrack}>
                <div style={{
                  ...styles.progressFill,
                  width: `${(batchProgress.done / Math.max(batchProgress.total, 1)) * 100}%`,
                }} />
              </div>
            </div>
          )}

          {batchResults.length > 0 && (
            <div style={styles.berryList}>
              {batchResults.map((r, i) => (
                <div key={i} style={styles.batchRow}>
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {r.filename}
                  </span>
                  {r.data ? (
                    <span style={{ color: r.data.summary.pct_rot > 20 ? "#ff5050" : "#50ff50" }}>
                      {r.data.summary.pct_rot}% rot
                    </span>
                  ) : (
                    <span style={{ color: "#ff8080", fontSize: 12 }}>{r.error}</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Single-image mode ──────────────────────────────────────────── */}
      {mode === "single" && (originalUrl || result) && (
        <div style={styles.layout}>

          {/* Image + toggle */}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>

            {/* Toggle buttons — only shown when results exist */}
            {result && (
              <div style={styles.toggleRow}>
                <button
                  style={{ ...styles.toggleBtn, ...(showAnnotated ? {} : styles.toggleActive) }}
                  onClick={() => setShowAnnotated(false)}
                >
                  Original
                </button>
                <button
                  style={{ ...styles.toggleBtn, ...(showAnnotated ? styles.toggleActive : {}) }}
                  onClick={() => setShowAnnotated(true)}
                >
                  Annotated
                </button>
              </div>
            )}

            <div style={{ position: "relative", display: "inline-block" }}>
              <img
                ref={imgRef}
                src={
                  showAnnotated && result
                    ? `data:image/png;base64,${result.annotated_image}`
                    : originalUrl
                }
                alt={showAnnotated ? "annotated" : "original"}
                style={{
                  ...styles.img,
                  cursor: showAnnotated && result ? "crosshair" : "default",
                  // Dim image while API is running
                  opacity: loading ? 0.5 : 1,
                  transition: "opacity 0.3s",
                }}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setHovered(null)}
              />

              {/* Loading overlay */}
              {loading && (
                <div style={styles.loadingOverlay}>
                  <span style={styles.loadingText}>Analysing…</span>
                </div>
              )}

              {/* Tooltip — only on annotated view */}
              {hoveredBerry && showAnnotated && (
                <div style={{
                  ...styles.tooltip,
                  left: tooltip.x + 14,
                  top:  tooltip.y - 10,
                }}>
                  <div style={{
                    fontWeight: "bold",
                    color: CLASS_COLOR[hoveredBerry.predicted_class],
                    marginBottom: 6,
                  }}>
                    Berry #{hoveredBerry.mask_index} — {CLASS_LABEL[hoveredBerry.predicted_class]}
                    {" "}({(hoveredBerry.confidence * 100).toFixed(1)}%)
                  </div>
                  <ProbBar label="Rot"  value={hoveredBerry.p_rot}  color="#ff5050" />
                  <ProbBar label="Ripe" value={hoveredBerry.p_ripe} color="#50ff50" />
                  <div style={{ fontSize: 11, color: "#aaa", marginTop: 6 }}>
                    SAM score: {(hoveredBerry.sam_score * 100).toFixed(1)}%
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Summary panel — only when results exist */}
          {result && (
            <div style={styles.summary}>
              <h2 style={{ marginTop: 0 }}>Summary</h2>
              <Stat label="Total detected" value={result.summary.total} />
              <Stat label="Ripe"  value={result.summary.n_ripe} color="#50ff50" />
              <Stat label="Rot"   value={result.summary.n_rot}  color="#ff5050" />
              <Stat label="% Rot" value={`${result.summary.pct_rot}%`}
                    color={result.summary.pct_rot > 20 ? "#ff5050" : "#50ff50"} />
              {result.detected_label && result.detected_label.type !== "none" && (
                <Stat
                  label={result.detected_label.type === "barcode" ? "Barcode" : "Detected label"}
                  value={result.detected_label.value}
                />
              )}

              <h3 style={{ marginTop: 24, marginBottom: 8 }}>Per-Berry</h3>
              <div style={styles.berryList}>
                {result.cranberries.map((c) => {
                  const isHovered = hovered === c.mask_index;
                  return (
                    <div
                      key={c.mask_index}
                      style={{
                        ...styles.berryRow,
                        borderLeft: `4px solid ${CLASS_COLOR[c.predicted_class]}`,
                        // Highlight matching berry when hovering image,
                        // or highlight on direct hover of this row
                        background: isHovered ? "#2a2a2a" : "transparent",
                        outline: isHovered
                          ? `1px solid ${CLASS_COLOR[c.predicted_class]}`
                          : "none",
                      }}
                      onMouseEnter={() => setHovered(c.mask_index)}
                      onMouseLeave={() => setHovered(null)}
                    >
                      <span style={{ color: CLASS_COLOR[c.predicted_class], fontWeight: isHovered ? "bold" : "normal" }}>
                        #{c.mask_index} {CLASS_LABEL[c.predicted_class]}
                      </span>
                      <span style={{ color: "#aaa", fontSize: 12 }}>
                        rot {(c.p_rot * 100).toFixed(0)}% /
                        ripe {(c.p_ripe * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Change access key */}
      <div style={{ textAlign: "center", marginTop: 32 }}>
        <button
          style={styles.changeKeyBtn}
          onClick={() => { clearStoredApiKey(); setApiKey(null); }}
        >
          Change access key
        </button>
      </div>

      {/* How-to modal */}
      {showHowTo && <HowToModal onClose={() => setShowHowTo(false)} />}
    </div>
  );
}

// ── Reusable components ────────────────────────────────────────────────────
function ProbBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 5 }}>
      <div style={{ display: "flex", justifyContent: "space-between",
                    fontSize: 12, marginBottom: 2 }}>
        <span>{label}</span>
        <span>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ background: "#333", borderRadius: 4, height: 8 }}>
        <div style={{
          width: `${value * 100}%`, height: "100%",
          background: color, borderRadius: 4,
          transition: "width 0.3s",
        }} />
      </div>
    </div>
  );
}

function Stat({ label, value, color = "white" }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between",
                  padding: "6px 0", borderBottom: "1px solid #2a2a2a" }}>
      <span style={{ color: "#aaa" }}>{label}</span>
      <span style={{ color, fontWeight: "bold", textAlign: "right", marginLeft: 12 }}>{value}</span>
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
  controls: {
    display: "flex", justifyContent: "center",
    alignItems: "center", gap: 12, marginBottom: 24,
  },
  uploadBtn: {
    display: "block", padding: "12px 32px",
    background: "#e63946", color: "white", borderRadius: 8,
    cursor: "pointer", fontWeight: "bold", fontSize: 16,
    userSelect: "none",
  },
  howToBtn: {
    padding: "10px 20px", background: "#222", color: "#aaa",
    border: "1px solid #444", borderRadius: 8,
    cursor: "pointer", fontSize: 14, fontWeight: "normal",
  },
  changeKeyBtn: {
    padding: "6px 14px", background: "transparent", color: "#666",
    border: "none", borderRadius: 6,
    cursor: "pointer", fontSize: 12, textDecoration: "underline",
  },
  errorBox: {
    maxWidth: 500, margin: "0 auto 24px",
    background: "#2a1010", border: "1px solid #ff5050",
    borderRadius: 8, padding: "12px 16px",
    color: "#ff8080", textAlign: "center",
  },
  layout: {
    display: "flex", gap: 24, alignItems: "flex-start",
    flexWrap: "wrap", justifyContent: "center",
  },
  toggleRow: {
    display: "flex", gap: 8, justifyContent: "center", marginBottom: 16,
  },
  toggleBtn: {
    padding: "6px 18px", background: "#222", color: "#aaa",
    border: "1px solid #444", borderRadius: 6,
    cursor: "pointer", fontSize: 13,
  },
  toggleActive: {
    background: "#333", color: "white", borderColor: "#888",
  },
  img: {
    maxWidth: "70vw", maxHeight: "80vh",
    borderRadius: 8, display: "block",
  },
  loadingOverlay: {
    position: "absolute", inset: 0,
    display: "flex", alignItems: "center", justifyContent: "center",
    borderRadius: 8,
  },
  loadingText: {
    background: "rgba(0,0,0,0.6)", padding: "8px 20px",
    borderRadius: 6, color: "white", fontSize: 15,
  },
  tooltip: {
    position: "fixed", zIndex: 1000,
    background: "#1e1e1e", border: "1px solid #444",
    borderRadius: 8, padding: "10px 14px",
    pointerEvents: "none", minWidth: 190,
    boxShadow: "0 4px 20px rgba(0,0,0,0.6)",
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
    fontSize: 13, transition: "background 0.15s, outline 0.15s",
  },
  batchBox: {
    maxWidth: 500, margin: "0 auto 24px",
    background: "#1a1a1a", borderRadius: 8, padding: 20,
  },
  progressTrack: {
    background: "#333", borderRadius: 4, height: 8, overflow: "hidden",
  },
  progressFill: {
    height: "100%", background: "#e63946", transition: "width 0.2s",
  },
  batchRow: {
    padding: "6px 10px", marginBottom: 4, borderRadius: 4,
    display: "flex", justifyContent: "space-between", alignItems: "center",
    gap: 10, fontSize: 13, background: "#222",
  },
  // ── Access key gate ─────────────────────────────────────────────────────
  keyGateBox: {
    maxWidth: 400, margin: "10vh auto 0",
    background: "#1a1a1a", border: "1px solid #333",
    borderRadius: 12, padding: 28,
  },
  keyInput: {
    width: "100%", padding: "10px 14px", fontSize: 15,
    background: "#111", border: "1px solid #444", borderRadius: 8,
    color: "white", boxSizing: "border-box",
  },
  // ── How-to modal ──────────────────────────────────────────────────────
  modalOverlay: {
    position: "fixed", inset: 0, zIndex: 2000,
    background: "rgba(0,0,0,0.75)",
    display: "flex", alignItems: "center", justifyContent: "center",
  },
  modalBox: {
    background: "#1a1a1a", border: "1px solid #333",
    borderRadius: 12, padding: 28, maxWidth: 480, width: "90%",
    boxShadow: "0 8px 40px rgba(0,0,0,0.8)",
  },
  modalText: {
    color: "#ccc", lineHeight: 1.6, marginBottom: 12, fontSize: 14,
  },
  closeBtn: {
    marginTop: 8, padding: "10px 28px",
    background: "#e63946", color: "white",
    border: "none", borderRadius: 8,
    cursor: "pointer", fontWeight: "bold", fontSize: 15,
    display: "block", marginLeft: "auto",
  },
};
