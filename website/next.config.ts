import type { NextConfig } from "next";

const repository = process.env.GITHUB_REPOSITORY?.split("/")[1] ?? "";
const isProjectPage = process.env.GITHUB_ACTIONS === "true" && repository && !repository.endsWith(".github.io");
const basePath = isProjectPage ? `/${repository}` : "";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  assetPrefix: basePath || undefined,
  images: { unoptimized: true },
};

export default nextConfig;
