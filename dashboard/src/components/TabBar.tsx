import { useRef } from "react";

export type TabDef = { id: string; label: string };

export default function TabBar(props: {
  tabs: TabDef[];
  active: string;
  onSelect: (id: string) => void;
}) {
  const { tabs, active, onSelect } = props;
  const btnRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKeyDown = (e: React.KeyboardEvent, index: number) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const dir = e.key === "ArrowRight" ? 1 : -1;
    const next = (index + dir + tabs.length) % tabs.length;
    onSelect(tabs[next].id);
    btnRefs.current[next]?.focus();
  };

  return (
    <div className="tabbar" role="tablist" aria-label="Dashboard sections">
      {tabs.map((tab, i) => {
        const isActive = tab.id === active;
        return (
          <button
            key={tab.id}
            ref={(el) => {
              btnRefs.current[i] = el;
            }}
            type="button"
            role="tab"
            aria-selected={isActive}
            tabIndex={isActive ? 0 : -1}
            className={`tab${isActive ? " active" : ""}`}
            onClick={() => onSelect(tab.id)}
            onKeyDown={(e) => onKeyDown(e, i)}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
