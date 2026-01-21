# AGENTS.md - Codebase Guide for Agentic Development

## Project Overview

This is a **static HTML/JavaScript blog** hosted on GitHub Pages. It's a minimal hand-maintained website with client-side dynamic content loading. No build system, framework, or testing setup - pure static files.

### Technology Stack
- **Languages**: HTML5, CSS3, JavaScript (ES6+)
- **External Library**: marked.js (CDN for markdown parsing)
- **Content Format**: Markdown (.md files) + JSON (metadata index)
- **Deployment**: GitHub Pages (static hosting)

---

## Build, Lint, Test Commands

**No build system, no linter, no tests.**

This is a simple static site. To work with it:
- **Run locally**: Open HTML files directly in a browser or use any static file server
- **Deploy**: Push to GitHub Pages branch (or use root `/` for user/org pages)
- **No single test command**: Manual browser testing only

---

## Code Style Guidelines

### File Naming Conventions
- **HTML files**: `kebab-case.html` (e.g., `pitfalls.html`, `view.html`)
- **Markdown files**: `camelCase.md` (e.g., `cuda13onFedora.md`)
- **JSON files**: `kebab-case.json` (e.g., `index.json`)
- **All lowercase**, no spaces, use hyphens for UI/config files, camelCase for content

### Directory Structure
```
/
├── index.html              # Main landing page
├── [section].html          # Section listing pages
└── [section]/
    ├── index.json          # Metadata index for section content
    ├── view.html           # Markdown viewer page
    └── article.md          # Individual markdown articles
```

### JavaScript Naming Patterns
- **Functions**: `camelCase` for async functions (`loadPitfalls()`, `renderPitfalls()`, `loadPitfall()`, `showError()`)
- **Variable names**: `camelCase`
- **Use async/await pattern** for all async operations
- **Always handle errors** with try-catch blocks

### CSS Naming Patterns
- **Classes**: `kebab-case` (e.g., `.pitfall-list`, `.link-card`, `.back-link`, `.content`)
- **IDs**: `camelCase` (e.g., `#pitfallList`, `#content`)
- **No shared stylesheets**: All CSS is inline in `<style>` tags within HTML files
- **Consistent theme**: Purple gradient (#667eea to #764ba2)
- **Typography**: System fonts stack (-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif)

### HTML Structure Patterns
- **Semantic HTML5**: Use `<header>`, `<section>`, `<nav>`, etc.
- **Viewport meta tag**: `<meta name="viewport" content="width=device-width, initial-scale=1.0">` (always include)
- **Language attribute**: `<html lang="zh-CN">` or appropriate language code
- **Self-contained pages**: Each HTML file includes its own CSS/JS

### Error Handling Patterns
```javascript
// Always use try-catch for async operations
async function loadData() {
    try {
        const response = await fetch('path/to/data.json');
        if (!response.ok) {
            throw new Error('File not found');
        }
        const data = await response.json();
        // process data
    } catch (error) {
        console.error('Failed to load data:', error);
        // Show user-friendly error message in DOM
        document.getElementById('container').innerHTML = `
            <div class="error-state">
                <h3>Error</h3>
                <p>${error.message}</p>
            </div>
        `;
    }
}
```

### Adding New Content
1. Create markdown file in appropriate section directory (e.g., `pitfalls/newArticle.md`)
2. Update section's `index.json` with metadata:
   ```json
   {
     "id": "uniqueId",
     "title": "Article Title",
     "date": "2026-01-21",
     "file": "newArticle.md"
   }
   ```
3. Follow existing markdown patterns (use fenced code blocks with language specified)
4. Test by opening `view.html?file=section/newArticle.md` in browser

### External Libraries
- **Use CDN links** in script tags when needed
- **Example**: `<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>`
- **No npm/yarn**: Add external deps via CDN only

### URL Parameter Handling
```javascript
// Parse query parameters
const params = new URLSearchParams(window.location.search);
const file = params.get('file');

// Check if parameter exists
if (!file) {
    showError('Missing required parameter');
    return;
}
```

### CSS Best Practices
- **Mobile-first**: Add media queries at bottom of CSS blocks
- **Flexbox/Grid**: Use for layouts
- **Box-sizing**: Set `* { box-sizing: border-box; }` at top of all styles
- **Reset**: Include basic margin/padding reset
- **Gradients**: Use `linear-gradient(135deg, #667eea 0%, #764ba2 100%)` for backgrounds
- **Shadows**: Consistent shadow depths: `0 10px 30px rgba(0, 0, 0, 0.1)` for cards
- **Transitions**: Use `transition: all 0.3s ease;` for hover effects
- **Border radius**: 12px for cards, 8px for smaller elements

### Code Comments
- **Chinese comments**: Content and UI are in Chinese, use Chinese for comments too
- **Keep it minimal**: Code is straightforward, don't over-comment

---

## Working with This Codebase

### When Adding a New Section
1. Create section directory (e.g., `articles/`)
2. Create `index.json` with empty array `[]`
3. Copy and modify `pitfalls.html` as `[section].html`
4. Copy `pitfalls/view.html` to section directory and update paths
5. Link from `index.html` if needed

### Common Tasks
- **Add article**: Create `.md` file, update `index.json`
- **Style change**: Edit inline `<style>` block in relevant HTML file
- **Change layout**: Modify HTML structure and corresponding CSS
- **Debug**: Use browser DevTools, check console for errors

### Git Workflow
- **Branching**: Simple workflow, no strict branching rules
- **Commit**: Clear commit messages in English or Chinese matching content
- **Deployment**: Push to GitHub Pages, automatic deployment

---

## Notes for Agents

- This is a **minimal static site** - don't over-engineer
- No type checking, no linter - you're responsible for maintaining consistency
- Follow existing patterns exactly (color scheme, layout, naming)
- Test changes manually in browser before completing
- If something seems wrong, ask - the patterns here are simple and intentional
