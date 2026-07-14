// frontend/src/models/capabilities.tsx
//
// Frontend access layer for the backend capability manifest (GET /capabilities,
// see models/capabilities.py). The wizard renders itself entirely from this data
// so adding a new process model requires no frontend changes — only a backend
// registry + manifest entry.
//
// Exposes:
//   * <CapabilitiesProvider>  — fetches the manifest once and shares it.
//   * useCapabilities()       — { models, loading, error, getModel }.
//   * pure helpers (defaults / validation) usable outside React.
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  getCapabilities,
  type ConfigConstraint,
  type ConfigFieldMeta,
  type ModelCapability,
} from "../lib/api";

export type ConfigValue = number | boolean;
export type ModelConfig = Record<string, ConfigValue>;

// -----------------------------------------------------------------------------
// Pure helpers (no React) — shared by the wizard for defaults + validation.
// -----------------------------------------------------------------------------

/** Build a config object of defaults from a model's field schema. */
export function defaultConfigFor(model: ModelCapability): ModelConfig {
  const out: ModelConfig = {};
  for (const f of model.config_fields) out[f.key] = f.default;
  return out;
}

/** Validate a single field value against its declarative rules. */
export function validateField(field: ConfigFieldMeta, value: ConfigValue): boolean {
  if (field.kind === "boolean") return typeof value === "boolean";

  if (typeof value !== "number" || !Number.isFinite(value)) return false;
  if (field.integer && !Number.isInteger(value)) return false;
  if (field.odd && (!Number.isInteger(value) || value % 2 !== 1)) return false;
  if (field.min != null && value < field.min) return false;
  if (field.max != null && value > field.max) return false;
  if (field.gt != null && value <= field.gt) return false;
  if (field.lt != null && value >= field.lt) return false;
  return true;
}

function checkConstraint(c: ConfigConstraint, cfg: ModelConfig): boolean {
  const left = cfg[c.left];
  const right = cfg[c.right];
  if (typeof left !== "number" || typeof right !== "number") return true;
  switch (c.type) {
    case "lte":
      return left <= right;
    default:
      return true;
  }
}

/** Validate a full config: every field rule + every cross-field constraint. */
export function validateConfig(model: ModelCapability, cfg: ModelConfig): boolean {
  for (const f of model.config_fields) {
    if (!validateField(f, cfg[f.key])) return false;
  }
  for (const c of model.config_constraints ?? []) {
    if (!checkConstraint(c, cfg)) return false;
  }
  return true;
}

/** First failing cross-field constraint message, if any (for inline UI hints). */
export function firstConstraintError(
  model: ModelCapability,
  cfg: ModelConfig
): string | null {
  for (const c of model.config_constraints ?? []) {
    if (!checkConstraint(c, cfg)) return c.message ?? "Invalid configuration.";
  }
  return null;
}

/** Whether an explainability method value is offered by a model. */
export function isExplainAllowed(
  model: ModelCapability | undefined,
  method: string | null
): boolean {
  if (!model || !method) return true;
  return model.explain_methods.some((m) => m.value === method);
}

// -----------------------------------------------------------------------------
// React context
// -----------------------------------------------------------------------------
type CapabilitiesState = {
  models: ModelCapability[];
  loading: boolean;
  error: string | null;
  getModel: (id: string | null | undefined) => ModelCapability | undefined;
  reload: () => void;
};

const CapabilitiesContext = createContext<CapabilitiesState | null>(null);

export function CapabilitiesProvider({ children }: { children: ReactNode }) {
  const [models, setModels] = useState<ModelCapability[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getCapabilities()
      .then((res) => {
        if (cancelled) return;
        setModels(res.models ?? []);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [nonce]);

  const value = useMemo<CapabilitiesState>(() => {
    const byId = new Map(models.map((m) => [m.id, m]));
    return {
      models,
      loading,
      error,
      getModel: (id) => (id ? byId.get(id) : undefined),
      reload: () => setNonce((n) => n + 1),
    };
  }, [models, loading, error]);

  return (
    <CapabilitiesContext.Provider value={value}>
      {children}
    </CapabilitiesContext.Provider>
  );
}

export function useCapabilities(): CapabilitiesState {
  const ctx = useContext(CapabilitiesContext);
  if (!ctx) {
    throw new Error("useCapabilities must be used within a CapabilitiesProvider");
  }
  return ctx;
}
