export default function PageHeader({ eyebrow, title, description, children }) {
  return (
    <section className="page-header">
      <div>
        {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {children ? <div className="page-header-side">{children}</div> : null}
    </section>
  );
}
