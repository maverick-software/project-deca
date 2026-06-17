import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = {
  /** Human label for the panel, shown in the fallback. */
  label?: string;
  /** When this value changes, a previously-caught error is cleared and the
   *  children are re-rendered. Pass the poll cycle so a transient bad frame
   *  self-heals on the next clean update. */
  resetKey?: unknown;
  children: ReactNode;
};

type State = { error: Error | null };

/**
 * Isolates a render crash to a single panel. A stray NaN/null from the server
 * (transient neural instability) shows a small recoverable card here instead of
 * blanking the whole dashboard.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`[panel:${this.props.label ?? "?"}]`, error, info.componentStack);
  }

  componentDidUpdate(prev: Props): void {
    if (this.state.error && prev.resetKey !== this.props.resetKey) {
      this.setState({ error: null });
    }
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="panel">
          <div className="empty">
            <b>{this.props.label ?? "This panel"}</b> hit a render error and was isolated so the
            rest of the dashboard keeps working. This is usually a momentary unstable (NaN) value
            from the agent and clears on the next clean update.
            <div style={{ marginTop: 8 }}>
              <button className="btn" onClick={() => this.setState({ error: null })}>
                Retry
              </button>
            </div>
            <pre
              style={{ marginTop: 8, opacity: 0.55, fontSize: 11, whiteSpace: "pre-wrap" }}
            >
              {String(this.state.error.message)}
            </pre>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
