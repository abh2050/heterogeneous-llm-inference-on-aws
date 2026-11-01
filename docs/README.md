# Inference Stack Atlas

Static GitHub Pages explainer for the Intel, NVIDIA, and AMD production LLM inference stacks. It includes a guided, dependency-ordered system designer that recommends components and exports the completed architecture as Markdown.

## Preview locally

```bash
python3 -m http.server 8080 --directory docs
```

Then open <http://localhost:8080>.

## Publish with GitHub Pages

In the repository settings, choose **Pages → Deploy from a branch**, select the default branch, and use the `/docs` folder. The site uses no build step or external runtime dependencies.
