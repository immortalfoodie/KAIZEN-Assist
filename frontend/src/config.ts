// Central API configuration for the frontend
const isLocalVite = typeof window !== "undefined" && window.location.port === "5173";
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (isLocalVite ? "http://localhost:8000" : (typeof window !== "undefined" ? window.location.origin : ""));
