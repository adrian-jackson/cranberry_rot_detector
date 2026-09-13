// apiKey.js
// Stores the shared access key in the browser (localStorage) instead of
// baking it into the Vite build — a VITE_-prefixed env var gets inlined
// into the public JS bundle, which anyone visiting the site could read
// out of devtools. Keeping it in localStorage means it's only ever known
// by people you've actually handed the key to.

const STORAGE_KEY = "cranberry_api_key";

export function getStoredApiKey() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function setStoredApiKey(key) {
  try {
    localStorage.setItem(STORAGE_KEY, key);
  } catch {
    // Storage disabled (e.g. private browsing) — key just won't persist.
  }
}

export function clearStoredApiKey() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
