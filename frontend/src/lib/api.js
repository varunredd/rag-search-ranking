const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function handleResponse(response) {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Request failed");
  }
  return response.json();
}

export async function getJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  return handleResponse(res);
}

export async function postJSON(path, payload) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}
