# FlagGems-vllm Documentation

This directory contains the documentation website for FlagGems-vllm, built with [Hugo](https://gohugo.io/) and the [hugo-book](https://github.com/alex-shpak/hugo-book) theme.

## Prerequisites

Install Hugo (extended version):

```bash
# macOS
brew install hugo

# Ubuntu/Debian
sudo apt install hugo

# Or download from https://github.com/gohugoio/hugo/releases
```

Verify installation:

```bash
hugo version
# Should show v0.120.0 or newer
```

## Building the Documentation

### Development Server

Run a local development server with live reload:

```bash
cd docs
hugo server
```

Then open http://localhost:1313/FlagGems-vllm/ in your browser.

### Build Static Site

Generate static HTML files:

```bash
cd docs
hugo
```

Output will be in `docs/public/`.

### Clean Build

```bash
cd docs
rm -rf public resources
hugo
```

## Documentation Structure

```
docs/
├── archetypes/         # Content templates
├── assets/             # Images, CSS, JS
├── content/            # Markdown content
│   ├── en/            # English content
│   │   ├── _index.md
│   │   ├── getting-started/
│   │   ├── usage/
│   │   ├── performance/
│   │   ├── references/
│   │   ├── testing/
│   │   └── contribution/
│   └── zh-cn/         # Chinese content
│       └── (same structure as en/)
├── i18n/              # Translations
│   ├── en.toml
│   └── zh-cn.toml
├── layouts/           # Custom layouts and shortcodes
│   └── shortcodes/
│       ├── operator-list.html
│       ├── benchmark-table.html
│       └── coverage-data.html
├── static/            # Static files (images, CSS, JS)
│   ├── css/
│   ├── js/
│   └── images/
├── themes/            # Hugo themes
│   └── hugo-book/
└── hugo.yaml          # Hugo configuration
```

## Adding Content

### Create a New Page

```bash
cd docs
hugo new content/en/my-section/my-page.md
```

Edit the frontmatter and content:

```markdown
---
title: My Page Title
weight: 10
---

# My Page Title

Content goes here.
```

### Frontmatter Fields

- `title`: Page title (required)
- `weight`: Ordering within section (lower = higher priority)
- `bookCollapseSection`: Set to `true` for section index pages
- `bookHidden`: Hide page from navigation
- `bookToC`: Enable/disable table of contents for this page

### Sections

To create a collapsible section, add an `_index.md` file:

```markdown
---
title: My Section
weight: 20
bookCollapseSection: true
---

# My Section

Section overview goes here.
```

## Shortcodes

### operator-list

Renders the operator list from `conf/operators.yaml`:

```markdown
{{< operator-list >}}
```

### benchmark-table

Renders benchmark results with Tabulator.js:

```markdown
---
title: Benchmark Results
useTabulator: true
---

{{< benchmark-table >}}
```

### coverage-data

Lists coverage reports from `static/coverage/`:

```markdown
{{< coverage-data >}}
```

## Languages

The site supports English (`en`) and Chinese (`zh-cn`).

### Add a Translation

1. Create the same page structure under `content/zh-cn/`
2. Translate the content
3. Ensure frontmatter `weight` matches for proper navigation

### Language Switching

Language switcher appears in the site header automatically.

## Deployment

### GitHub Pages

The site can be deployed to GitHub Pages:

1. Build the site: `hugo`
2. Push `docs/public/` to the `gh-pages` branch
3. Configure GitHub Pages to serve from the `gh-pages` branch

Or use GitHub Actions (see `.github/workflows/docs.yml` if added).

### Custom Domain

Set `baseURL` in `hugo.yaml`:

```yaml
baseURL: https://your-domain.com/FlagGems-vllm/
```

## Updating Operator Metadata

The operator list is generated from `conf/operators.yaml` at the repository root.

After updating `conf/operators.yaml`:

1. Rebuild the docs: `hugo`
2. Check the operator list page: http://localhost:1313/FlagGems-vllm/references/operators/

## Updating Benchmark Data

Benchmark results are mounted from `benchmark/` directory (configured in `hugo.yaml`).

After running benchmarks with `--record --output results.json`, the data becomes available to the `benchmark-table` shortcode.

## Troubleshooting

### Hugo Not Found

Install Hugo extended version from https://github.com/gohugoio/hugo/releases

### Theme Not Loading

Ensure `themes/hugo-book` exists:

```bash
ls docs/themes/hugo-book
```

If missing, copy from FlagGems reference or clone:

```bash
cd docs/themes
git clone https://github.com/alex-shpak/hugo-book
```

### Broken Links

Check for broken internal links:

```bash
cd docs
hugo server
# Visit site and check browser console for 404s
```

Use absolute paths for cross-references:

```markdown
[Link text](/FlagGems-vllm/section/page/)
```

### Port Already in Use

Change the port:

```bash
hugo server --port 1314
```

## Contributing

When adding documentation:

1. Follow the existing structure
2. Use clear, concise language
3. Add code examples where helpful
4. Test links and formatting locally before committing
5. Keep English and Chinese versions in sync (if translating)

## License

Documentation is licensed under Apache License 2.0, same as the project code.
