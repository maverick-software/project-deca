import { useEffect, useRef, useState } from "react";

/** Poll an async fetcher on an interval; keeps last good value and error state. */
export function usePolling<T>(
  fn: () => Promise<T>,
  intervalMs: number,
  deps: unknown[] = [],
): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      try {
        const v = await fnRef.current();
        if (alive) {
          setData(v);
          setError(null);
        }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      }
      if (alive) timer = setTimeout(tick, intervalMs);
    };
    void tick();

    return () => {
      alive = false;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, ...deps]);

  return { data, error };
}

export type HistorySample = {
  t: number;
  pcLoss: number;
  viability: number;
  pain: number;
  pleasure: number;
  fwdErr?: number;
  assistGain?: number;
};

/** Rolling client-side history for time-series charts. */
export function useHistory(
  sample: HistorySample | null,
  cap = 240,
): HistorySample[] {
  const [history, setHistory] = useState<HistorySample[]>([]);
  const lastT = useRef<number>(-1);

  useEffect(() => {
    if (!sample || sample.t === lastT.current) return;
    lastT.current = sample.t;
    setHistory((h) => {
      const next = [...h, sample];
      return next.length > cap ? next.slice(next.length - cap) : next;
    });
  }, [sample, cap]);

  return history;
}
