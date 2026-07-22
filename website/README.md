# KuaiLive-M3 Website

Official project website for **KuaiLive-M3: A Multi-Modal, Multi-Domain, and Multi-Feedback Dataset for Live Streaming Recommendation**.

The site contains:

- dataset overview and research positioning;
- headline statistics;
- release directory structure;
- searchable field-level documentation for all 18 tables and collections;
- official benchmark coverage for cross-domain recommendation, live-stream highlight prediction, and questionnaire-based recommendation;
- direct access to the dataset on Hugging Face.

## Configure the benchmark link

After publishing the benchmark repository, replace the placeholder URL in:

```text
app/site-config.ts
```

This single value controls all benchmark GitHub links in the header, benchmark section, download banner, and footer.

## Local preview

```bash
npm install
npm run dev
```

Open <http://localhost:3000>.

## Production build

```bash
npm run build:github
```

The static site is exported to `out/`.

## Deploy on GitHub Pages

1. Create a GitHub repository and push the entire KLM3 repository to its
   `main` branch.
2. In **Settings → Pages**, choose **GitHub Actions** as the source.
3. The workflow at `../.github/workflows/deploy-pages.yml` builds this
   subdirectory and publishes the site automatically.

The configuration automatically adds the repository name as the base path for project pages such as `https://username.github.io/repository/`. User or organization pages ending in `.github.io` are served from the root path.

## Dataset

[KuaiLive-M3 on Hugging Face](https://huggingface.co/datasets/imgkkk2004/KuaiLive-M3)
