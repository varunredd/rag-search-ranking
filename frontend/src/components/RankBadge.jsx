export default function RankBadge({ delta = 0 }) {
  let label = "No change";
  let cls = "rank-neutral";
  if (delta > 0) {
    label = `Moved up +${delta}`;
    cls = "rank-up";
  } else if (delta < 0) {
    label = `Dropped ${delta}`;
    cls = "rank-down";
  }
  return <span className={`rank-badge ${cls}`}>{label}</span>;
}
