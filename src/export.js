// export.js
// Bundles per-image results into a single .zip download: one annotated PNG
// per image (under annotated/) plus a results.xlsx summarizing all of them.
// A zip (rather than the File System Access API) works the same way in
// every browser, not just Chrome/Edge.

import * as XLSX from "xlsx";
import JSZip from "jszip";

const COLUMNS = [
  "filename",
  "label",
  "pct_rot",
  "pct_ripe",
  "avg_dino_confidence",
  "detected_label",
  "annotated_image_link",
  "error",
];

function base64ToUint8Array(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function avgConfidence(cranberries) {
  if (!cranberries || cranberries.length === 0) return "";
  const sum = cranberries.reduce((acc, c) => acc + c.confidence, 0);
  return Math.round((sum / cranberries.length) * 1000) / 1000;
}

// results: array of { filename, data, error }
//   data  — the /predict JSON response, or null on failure
//   error — a human-readable failure message, or null on success
export async function exportResultsZip(results, zipFilename = "cranberry_results.zip") {
  const zip = new JSZip();
  const annotatedFolder = zip.folder("annotated");
  const rows = [];

  for (const { filename, data, error } of results) {
    const stem = filename.replace(/\.[^./\\]+$/, "");
    const imageName = `${stem}.png`;
    const hasImage = data && data.annotated_image;

    if (hasImage) {
      annotatedFolder.file(imageName, base64ToUint8Array(data.annotated_image));
    }

    rows.push({
      filename,
      label: data ? (data.summary.pct_rot > 50 ? "rot" : "ripe") : "",
      pct_rot: data ? data.summary.pct_rot : "",
      pct_ripe: data ? Math.round((100 - data.summary.pct_rot) * 10) / 10 : "",
      avg_dino_confidence: data ? avgConfidence(data.cranberries) : "",
      detected_label: data?.detected_label?.value || "",
      annotated_image_link: hasImage ? `annotated/${imageName}` : "",
      error: error || "",
    });
  }

  const worksheet = XLSX.utils.json_to_sheet(rows, { header: COLUMNS });
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, "Results");
  const xlsxBuffer = XLSX.write(workbook, { type: "array", bookType: "xlsx" });
  zip.file("results.xlsx", xlsxBuffer);

  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = zipFilename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
