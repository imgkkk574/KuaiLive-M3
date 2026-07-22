import type { Metadata } from "next";
import "./globals.css";

const [githubOwner, githubRepository] = (process.env.GITHUB_REPOSITORY ?? "imgkkk574/KuaiLive-M3").split("/");
const githubBase = githubRepository?.endsWith(".github.io")
  ? `https://${githubOwner}.github.io/`
  : `https://${githubOwner}.github.io/${githubRepository}/`;

const themeScript = `(() => {
  try {
    const stored = localStorage.getItem("klm3-theme");
    const dark = stored ? stored === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  } catch (_) {}
})();`;

export const metadata: Metadata = {
  metadataBase: new URL(githubBase),
  title: "KuaiLive-M3 | Live Streaming Recommendation Dataset",
  description: "A multi-modal, multi-domain, and multi-feedback dataset for live streaming recommendation, collected from Kuaishou.",
  keywords: ["KuaiLive-M3", "live streaming recommendation", "multimodal recommendation", "cross-domain recommendation", "Kuaishou dataset"],
  openGraph: {
    title: "KuaiLive-M3",
    description: "Multi-Modal · Multi-Domain · Multi-Feedback — a live streaming recommendation dataset from Kuaishou.",
    type: "website",
    images: [{ url: "og.png", width: 1200, height: 630, alt: "KuaiLive-M3 dataset" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "KuaiLive-M3",
    description: "A multi-modal, multi-domain, and multi-feedback dataset for live streaming recommendation.",
    images: ["og.png"],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: themeScript }} /></head>
      <body>{children}</body>
    </html>
  );
}
