import Link from "next/link";
import { BENCHMARK_GITHUB_URL, DATASET_URL } from "./site-config";
import { ThemeToggle } from "./theme-toggle";

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="container nav-inner">
        <Link className="brand" href="/" aria-label="KuaiLive-M3 home">
          <span className="brand-mark">M3</span>
          <span><b>KuaiLive-M3</b><small>Open Research Dataset</small></span>
        </Link>
        <nav aria-label="Primary navigation">
          <Link href="/#overview">Overview</Link>
          <Link href="/#statistics">Statistics</Link>
          <Link href="/#structure">Structure</Link>
          <Link href="/#benchmark">Benchmark</Link>
          <Link href="/description/">Descriptions</Link>
        </nav>
        <div className="nav-actions">
          <ThemeToggle />
          <a className="nav-download" href={DATASET_URL} target="_blank" rel="noreferrer">Dataset <span>↗</span></a>
        </div>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="container footer-grid">
        <div><div className="brand footer-brand"><span className="brand-mark">M3</span><span><b>KuaiLive-M3</b><small>Live recommendation, in motion.</small></span></div></div>
        <div><p className="footer-label">Navigate</p><Link href="/#overview">Overview</Link><Link href="/#statistics">Statistics</Link><Link href="/#benchmark">Benchmark</Link><Link href="/description/">Field descriptions</Link></div>
        <div><p className="footer-label">Resources</p><a href={DATASET_URL} target="_blank" rel="noreferrer">Hugging Face ↗</a><a href={BENCHMARK_GITHUB_URL} target="_blank" rel="noreferrer">Benchmark GitHub ↗</a><a href="https://www.kuaishou.com" target="_blank" rel="noreferrer">Kuaishou ↗</a></div>
      </div>
      <div className="container footer-bottom"><span>© 2026 KuaiLive-M3</span><span>For academic research use.</span></div>
    </footer>
  );
}
