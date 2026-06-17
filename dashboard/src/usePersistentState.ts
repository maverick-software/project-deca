import { useEffect, useRef, useState } from "react";

/**
 * useState whose value is mirrored to localStorage, so view preferences (camera
 * angle, sweep speed, active view, etc.) survive tab unmounts and page reloads.
 * Values are JSON-serialized, so strings, numbers, booleans, and plain objects
 * all work.
 */
export function usePersistentState<T>(
  key: string,
  initial: T,
): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw != null ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });

  // Persist on change. Skip the very first run so we don't rewrite the value we
  // just read (and so a changed key picks up its own stored value cleanly).
  const firstRun = useRef(true);
  useEffect(() => {
    if (firstRun.current) {
      firstRun.current = false;
      return;
    }
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // ignore quota / serialization errors
    }
  }, [key, value]);

  return [value, setValue];
}
