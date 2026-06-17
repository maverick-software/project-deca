/** Small "?" badge that reveals an explanatory tooltip on hover/focus. */
export default function Info(props: { tip: string; align?: "left" | "right" }) {
  return (
    <span className={`info ${props.align ?? "left"}`} tabIndex={0}>
      ?
      <span className="info-tip">{props.tip}</span>
    </span>
  );
}
