import Link from "next/link";
import { SiteFooter, SiteHeader } from "./site-chrome";
import { BENCHMARK_GITHUB_URL, DATASET_URL } from "./site-config";

const stats = [
  { value: "21,938", label: "Users" },
  { value: "35M", label: "Live interactions" },
  { value: "111M", label: "Short-video interactions" },
  { value: "88M", label: "Live segment embeddings" },
  { value: "6.56M", label: "Live rooms" },
  { value: "6.74M", label: "Short videos" },
  { value: "25K", label: "Questionnaire responses" },
  { value: "18", label: "Data tables & collections" },
];

const pillars = [
  {
    index: "01",
    eyebrow: "Multi-modal",
    title: "Content that evolves with the room",
    text: "Timestamped 128-dimensional segment embeddings capture how live content changes over time, complemented by room-level live representations and video-level short-video embeddings.",
    accent: "cyan",
  },
  {
    index: "02",
    eyebrow: "Multi-domain",
    title: "One audience, two connected domains",
    text: "Shared, anonymized users and creators connect detailed live-stream and short-video histories, enabling realistic cross-domain transfer and recommendation studies.",
    accent: "violet",
  },
  {
    index: "03",
    eyebrow: "Multi-feedback",
    title: "Behavior meets explicit judgment",
    text: "Rich implicit actions—watching, liking, commenting, gifting, following, sharing—are paired with real questionnaire-based feedback from live-stream viewers.",
    accent: "coral",
  },
];

const tree = [
  { name: "author_profile.csv", note: "creator profiles" },
  { name: "user_id_set.csv", note: "anonymized user universe" },
  { name: "live/", note: "live-streaming domain", folder: true },
  { name: "live_interaction.csv", note: "watch sessions & actions", depth: 1 },
  { name: "live_show.parquet", note: "feed impressions", depth: 1 },
  { name: "live_comment.csv", note: "timestamped comments", depth: 1 },
  { name: "live_like.csv · live_share.csv", note: "fine-grained events", depth: 1 },
  { name: "live_questionnaire.csv", note: "explicit feedback", depth: 1 },
  { name: "live_room_meta.parquet", note: "room lifecycle & metadata", depth: 1 },
  { name: "live_emb_64.parquet", note: "room-level embeddings", depth: 1 },
  { name: "live_emb_128_ts/", note: "18 segment-embedding shards", depth: 1, folder: true },
  { name: "short_video/", note: "short-video domain", folder: true },
  { name: "photo_interaction.csv", note: "daily behavior aggregates", depth: 1 },
  { name: "photo_play.parquet", note: "event-level playback", depth: 1 },
  { name: "photo_meta.parquet · photo_tag.csv", note: "metadata & taxonomy", depth: 1 },
  { name: "photo_emb_128.parquet", note: "video-level embeddings", depth: 1 },
];

export default function Home() {
  return (
    <main>
      <SiteHeader />

      <section className="hero" id="overview">
        <div className="hero-grid" aria-hidden="true" />
        <div className="hero-orb hero-orb-one" aria-hidden="true" />
        <div className="hero-orb hero-orb-two" aria-hidden="true" />
        <div className="container hero-inner">
          <div className="hero-copy">
            <div className="status-pill"><span /> Open dataset · Kuaishou</div>
            <p className="kicker">Live streaming recommendation, in motion</p>
            <h1>KuaiLive-<span>M3</span></h1>
            <p className="subtitle">A Multi-Modal, Multi-Domain, and Multi-Feedback Dataset for Live Streaming Recommendation</p>
            <p className="hero-description">
              KuaiLive-M3 connects temporally evolving live content, short-video behaviors,
              and questionnaire-based feedback in one large-scale, real-world benchmark.
            </p>
            <div className="hero-actions">
              <a className="button button-primary" href={DATASET_URL} target="_blank" rel="noreferrer">
                Explore on Hugging Face <span aria-hidden="true">↗</span>
              </a>
              <Link className="button button-secondary" href="/description/">
                Browse field descriptions <span aria-hidden="true">→</span>
              </Link>
              <a className="button button-secondary" href={BENCHMARK_GITHUB_URL} target="_blank" rel="noreferrer">
                Benchmark code <span aria-hidden="true">↗</span>
              </a>
            </div>
          </div>

          <div className="hero-signal" aria-label="The three dimensions of KuaiLive-M3">
            <div className="signal-header"><span>Dataset signal map</span><span className="signal-live">LIVE</span></div>
            <div className="signal-center">
              <div className="signal-ring ring-three" />
              <div className="signal-ring ring-two" />
              <div className="signal-ring ring-one" />
              <div className="signal-core">M3</div>
              <div className="signal-node node-modal"><b>88M</b><span>segments</span></div>
              <div className="signal-node node-domain"><b>2</b><span>domains</span></div>
              <div className="signal-node node-feedback"><b>25K</b><span>surveys</span></div>
            </div>
            <div className="signal-legend">
              <span><i className="dot cyan" /> Multi-modal</span>
              <span><i className="dot violet" /> Multi-domain</span>
              <span><i className="dot coral" /> Multi-feedback</span>
            </div>
          </div>
        </div>
      </section>

      <section className="section section-light" id="why-m3">
        <div className="container">
          <div className="section-heading split-heading">
            <div>
              <p className="eyebrow">Why KuaiLive-M3</p>
              <h2>Three dimensions.<br />One living ecosystem.</h2>
            </div>
            <p>Designed to close three long-standing gaps in public live-streaming datasets: evolving multimodal content, connected cross-domain behavior, and explicit user feedback.</p>
          </div>
          <div className="pillar-grid">
            {pillars.map((pillar) => (
              <article className={`pillar-card ${pillar.accent}`} key={pillar.index}>
                <div className="pillar-top"><span>{pillar.index}</span><i /></div>
                <p className="pillar-eyebrow">{pillar.eyebrow}</p>
                <h3>{pillar.title}</h3>
                <p>{pillar.text}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="section statistics" id="statistics">
        <div className="container">
          <div className="section-heading statistics-heading">
            <div>
              <p className="eyebrow">At a glance</p>
              <h2>Scale for real-world questions</h2>
            </div>
            <p>Fine-grained records across both domains, anchored by a shared user population and temporally aligned live content.</p>
          </div>
          <div className="stats-grid">
            {stats.map((stat, index) => (
              <div className="stat" key={stat.label}>
                <span className="stat-index">{String(index + 1).padStart(2, "0")}</span>
                <strong>{stat.value}</strong>
                <span>{stat.label}</span>
              </div>
            ))}
          </div>
          <p className="stat-note">Rounded counts are provided for readability. Refer to the released files for exact row counts.</p>
        </div>
      </section>

      <section className="section section-light" id="structure">
        <div className="container structure-layout">
          <div className="structure-copy">
            <p className="eyebrow">Dataset structure</p>
            <h2>Organized around two domains and one shared identity space</h2>
            <p>
              IDs are anonymized and remapped to consecutive integers. Live-stream files preserve room lifecycles and millisecond-level behaviors; short-video files include both aggregate and event-level playback records.
            </p>
            <div className="structure-callout">
              <span className="callout-mark">i</span>
              <p><b>Segment timing.</b> Each segment embedding is paired with its end timestamp, allowing alignment with watch sessions and other timestamped events.</p>
            </div>
            <Link className="text-link" href="/description/">View all 18 tables and fields <span>→</span></Link>
          </div>
          <div className="file-tree" role="region" aria-label="KuaiLive-M3 directory structure">
            <div className="tree-toolbar"><span className="tree-dots"><i /><i /><i /></span><b>KuaiLive-M3/</b><span>dataset root</span></div>
            <div className="tree-body">
              {tree.map((item, index) => (
                <div className={`tree-row depth-${item.depth ?? 0}`} key={`${item.name}-${index}`}>
                  <span className={`tree-icon ${item.folder ? "folder" : "file"}`}>{item.folder ? "▸" : "·"}</span>
                  <code>{item.name}</code>
                  <span>{item.note}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="section benchmark" id="benchmark">
        <div className="container">
          <div className="section-heading benchmark-heading">
            <div>
              <p className="eyebrow">Official benchmark</p>
              <h2>Three tasks for the three dimensions of M3</h2>
            </div>
            <div className="benchmark-intro">
              <p>The benchmark repository provides task-specific preprocessing, standardized evaluation protocols, representative baselines, and reproducible experiment runners.</p>
              <a className="text-link" href={BENCHMARK_GITHUB_URL} target="_blank" rel="noreferrer">View benchmark on GitHub <span>↗</span></a>
            </div>
          </div>

          <div className="benchmark-grid">
            <article className="benchmark-card benchmark-cdr">
              <div className="benchmark-card-top"><span className="benchmark-number">01</span><span className="benchmark-dimension">Multi-domain</span></div>
              <h3>Cross-domain recommendation</h3>
              <p className="benchmark-summary">Transfer preferences from short-video viewing to live-stream author recommendation over a shared user population.</p>
              <dl>
                <div><dt>Protocol</dt><dd>Target-domain chronological 80/10/10 split; source domain for training</dd></div>
                <div><dt>Metrics</dt><dd>Recall and NDCG @ 10, 20, 40</dd></div>
                <div><dt>Models</dt><dd>BPR, SASRec, LightGCN + 12 CDR baselines</dd></div>
              </dl>
              <div className="model-tags"><span>CMF</span><span>CoNet</span><span>EMCDR</span><span>DisenCDR</span><span>MGCCDR</span><span>+7</span></div>
            </article>

            <article className="benchmark-card benchmark-highlight">
              <div className="benchmark-card-top"><span className="benchmark-number">02</span><span className="benchmark-dimension">Multi-modal</span></div>
              <h3>Live-stream highlight prediction</h3>
              <p className="benchmark-summary">Rank engaging segments using timestamped content embeddings and causally aligned retention and engagement signals.</p>
              <dl>
                <div><dt>Protocol</dt><dd>Room-level chronological 70/10/20 split; online and offline settings</dd></div>
                <div><dt>Metrics</dt><dd>mAP, F1@50%, Kendall's tau, Spearman's rho</dd></div>
                <div><dt>Inputs</dt><dd>Embedding, causal statistics, and embedding + statistics</dd></div>
              </dl>
              <div className="model-tags"><span>MLP</span><span>GRU</span><span>Causal Transformer</span><span>Hierarchical Transformer</span></div>
            </article>

            <article className="benchmark-card benchmark-survey">
              <div className="benchmark-card-top"><span className="benchmark-number">03</span><span className="benchmark-dimension">Multi-feedback</span></div>
              <h3>Questionnaire-based recommendation</h3>
              <p className="benchmark-summary">Complement click histories with satisfied and dissatisfied questionnaire signals while preventing answer leakage.</p>
              <dl>
                <div><dt>Protocol</dt><dd>Author-level 5-core with chronological leave-one-out evaluation</dd></div>
                <div><dt>Metrics</dt><dd>MRR, HR, and NDCG over 1 positive + 99 negatives</dd></div>
                <div><dt>Models</dt><dd>Click-only, multi-behavior, and satisfaction-aware methods</dd></div>
              </dl>
              <div className="model-tags"><span>SASRec</span><span>FeedRec</span><span>DFN</span><span>DMT</span><span>SAQRec</span><span>+8</span></div>
            </article>
          </div>

          <div className="benchmark-footer">
            <div><span className="benchmark-footer-label">Repository contents</span><p>Shared KLM3 loader · independent preprocessing pipelines · tuning scripts · result parsers · reproducibility documentation</p></div>
            <a className="button benchmark-button" href={BENCHMARK_GITHUB_URL} target="_blank" rel="noreferrer">Benchmark code on GitHub <span>↗</span></a>
          </div>
        </div>
      </section>

      <section className="download-band" id="download">
        <div className="container download-inner">
          <div><p className="eyebrow">Open access</p><h2>Start exploring KuaiLive-M3</h2><p>Dataset files are hosted on Hugging Face. Field-level documentation is available on this site.</p></div>
          <div className="download-actions">
            <a className="button button-light" href={DATASET_URL} target="_blank" rel="noreferrer">Open dataset ↗</a>
            <a className="button button-outline-light" href={BENCHMARK_GITHUB_URL} target="_blank" rel="noreferrer">Benchmark code ↗</a>
          </div>
        </div>
      </section>

      <SiteFooter />
    </main>
  );
}
