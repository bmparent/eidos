# Review in progress

Open index.html in a browser. No build step, API key, CDN, or subscription is required. Upload all files together to a static host.

## Included
- index.html, styles.css, tokens.css, script.js: the same page renderer used by the preview.
- project.json: editable source, including embedded images; import into Eidos Playground.
- approved-glass-preset.json: unchanged source reference.
- assets/: locally packaged uploaded images. No remote image dependencies.
- AI-HANDOFF.md: design-specific extension instructions.

## Before publishing
- Replace unresolved links and sample destinations.
- Contact links are not a connected form. Review the destination and test it.
- Review links, keyboard navigation, and actual mobile devices.
- Contact links open the visitor’s email app; this package does not include a form backend.
- Fonts are system Arial/Helvetica and Georgia; no font license files are required.
- Built-in abstract SVG studies are included. You must have rights to uploaded images and supplied brand content.

## Glass
Renderer: eidos-original-glass-2.0.0. New projects use the original Eidos SVG optics and edge lighting shared with the site. Legacy projects retain their portable renderer until explicitly upgraded in Playground. Browser rendering varies. Reduced motion and forced colors have fallbacks; no JavaScript leaves a solid header.

## Integration
This is a complete standalone page, not a platform-specific embed. Existing site CSS and script policies require separate integration. Do not paste the whole document into a CMS content field.
