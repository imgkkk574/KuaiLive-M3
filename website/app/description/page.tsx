import type { Metadata } from "next";
import Link from "next/link";
import { dataTables } from "../dataset-fields";
import { SiteFooter, SiteHeader } from "../site-chrome";
import { FieldCatalog } from "./field-catalog";

export const metadata: Metadata = {
  title: "Field Descriptions | KuaiLive-M3",
  description: "Complete field-level documentation for the KuaiLive-M3 live streaming recommendation dataset.",
};

export default function DescriptionPage() {
  return (
    <main>
      <SiteHeader />
      <section className="docs-hero">
        <div className="docs-grid" aria-hidden="true" />
        <div className="container docs-hero-inner">
          <div>
            <p className="eyebrow">Dataset documentation</p>
            <h1>Field descriptions</h1>
            <p>A complete reference for KuaiLive-M3&apos;s shared entities, live-streaming records, short-video behaviors, metadata, and multimodal embeddings.</p>
          </div>
          <div className="docs-summary">
            <div><strong>18</strong><span>tables & collections</span></div>
            <div><strong>{dataTables.reduce((sum, table) => sum + table.fields.length, 0)}</strong><span>documented fields</span></div>
            <div><strong>2</strong><span>content domains</span></div>
          </div>
        </div>
      </section>

      <section className="docs-intro">
        <div className="container docs-intro-grid">
          <div className="docs-note"><span>Privacy</span><p>All user, content, creator, room, and segment identifiers are anonymized. Entity IDs are remapped to consecutive integers starting from 1 unless otherwise noted.</p></div>
          <div className="docs-links"><Link href="/">← Back to overview</Link><a href="https://huggingface.co/imgkkk2004/KuaiLive-M3/tree/main" target="_blank" rel="noreferrer">Open dataset ↗</a></div>
        </div>
      </section>

      <section className="catalog-section">
        <div className="container"><FieldCatalog tables={dataTables} /></div>
      </section>
      <SiteFooter />
    </main>
  );
}
