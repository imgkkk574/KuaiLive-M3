"use client";

import { useMemo, useState } from "react";
import type { DataTable } from "../dataset-fields";

const domains = ["All", "Shared", "Live", "Short video"] as const;

export function FieldCatalog({ tables }: { tables: DataTable[] }) {
  const [query, setQuery] = useState("");
  const [domain, setDomain] = useState<(typeof domains)[number]>("All");

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return tables.filter((table) => {
      const domainMatch = domain === "All" || table.domain === domain;
      const textMatch = !needle || [table.file, table.title, table.summary, table.note ?? "", ...table.fields.flatMap((field) => [field.name, field.type, field.description])]
        .join(" ").toLowerCase().includes(needle);
      return domainMatch && textMatch;
    });
  }, [domain, query, tables]);

  return (
    <>
      <div className="catalog-toolbar">
        <label className="search-box">
          <span aria-hidden="true">⌕</span>
          <span className="sr-only">Search fields</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search files, fields, or descriptions…" />
        </label>
        <div className="filter-tabs" aria-label="Filter by domain">
          {domains.map((item) => (
            <button key={item} className={domain === item ? "active" : ""} onClick={() => setDomain(item)}>{item}</button>
          ))}
        </div>
      </div>

      <div className="catalog-count"><span>{filtered.length}</span> of {tables.length} tables shown</div>

      {filtered.length ? (
        <div className="catalog-list">
          {filtered.map((table, index) => (
            <article className="field-card" id={table.id} key={table.id}>
              <div className="field-card-head">
                <div className="field-number">{String(index + 1).padStart(2, "0")}</div>
                <div className="field-title"><span className={`domain-tag ${table.domain.toLowerCase().replace(" ", "-")}`}>{table.domain}</span><h2>{table.file}</h2><p>{table.summary}</p></div>
                <div className="field-count">{table.fields.length}<span>fields</span></div>
              </div>
              {table.note && <div className="field-note"><b>Note</b><span>{table.note}</span></div>}
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
                  <tbody>
                    {table.fields.map((field) => (
                      <tr key={field.name}><td><code>{field.name}</code></td><td><span className="type-pill">{field.type}</span></td><td>{field.description}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className="empty-state"><strong>No matching fields</strong><p>Try a broader search or choose a different domain.</p></div>
      )}
    </>
  );
}

