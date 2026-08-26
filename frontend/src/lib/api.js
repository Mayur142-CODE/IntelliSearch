const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, { signal } = {}) {
  const res = await fetch(`${BASE_URL}${path}`, { signal });
  if (!res.ok) {
    throw new Error(`Request failed (${res.status})`);
  }
  return res.json();
}

export function searchProducts(query, { signal, limit = 10 } = {}) {
  return request(`/search?q=${encodeURIComponent(query)}&limit=${limit}`, { signal });
}

export function fetchSuggestions(query, { signal, limit = 8 } = {}) {
  return request(`/autocomplete?q=${encodeURIComponent(query)}&limit=${limit}`, { signal });
}

export function getProduct(id, { signal } = {}) {
  return request(`/api/products/${id}`, { signal });
}

export function getHealth({ signal } = {}) {
  return request(`/health`, { signal });
}

export { BASE_URL };
