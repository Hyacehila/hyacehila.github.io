# Hyacehila - Personal Portfolio & Blog

## 🌟 About

Welcome to my personal website! This is a modern, responsive website that integrates a personal portfolio with a technical blog, featuring bilingual support (English/Chinese).

The website is built with HTML, CSS, JavaScript, and powered by Jekyll for blog functionality.

- **Portfolio**: Showcases my education, work experience, projects, and skills
- **Blog**: Technical notes, project reviews, and thoughts

This project is forked from [personal-portfolio](https://github.com/ivansaul/personal-portfolio) and extensively modified to add Jekyll blog functionality.

## 🚀 Features

### Website Features
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Bilingual Support**: Switch between English and Chinese languages
- **Modern UI**: Clean, professional design with smooth animations
- **Interactive Navigation**: Smooth page transitions and user-friendly interface
- **Contact Form**: Functional contact section with validation

### Blog Features
- **Markdown Writing**: Write blog posts using Markdown syntax
- **Code Highlighting**: Syntax highlighting powered by Rouge (Monokai theme)
- **Math Formulas**: LaTeX math formula rendering via MathJax
- **Diagram Support**: Mermaid diagrams (flowcharts, sequence diagrams, Gantt charts, etc.)
- **Categories & Tags**: Organize articles with categories and tags
- **Article Series**: Support for series organization
- **Article Navigation**: Previous/Next article navigation
- **Draft System**: Draft preview and management support

## 📁 Project Structure

```
PersonelPage/
├── index.html              # Main page (portfolio)
├── _config.yml             # Jekyll configuration file
├── _posts/                 # Blog posts directory
│   └── YYYY-MM-DD-title.md # Posts named by date
├── _drafts/                # Blog drafts directory
├── _layouts/               # Jekyll layout templates
│   └── blog-post.html      # Blog post layout
├── _includes/              # Jekyll reusable components
│   ├── head.html
│   ├── footer.html
│   ├── mathjax.html        # Math formula support
│   └── mermaid.html        # Diagram support
├── assets/
│   ├── css/
│   │   └── style.css       # Main stylesheet
│   ├── js/
│   │   └── script.js       # JavaScript functionality
│   ├── gitbook/            # GitBook style resources
│   └── images/             # Images and icons
├── blog/                   # Blog-related pages
├── code/                   # Code examples directory
├── README.md               # This file
└── LICENSE                 # License information
```

## 🛠️ Technologies Used

### Core Technologies
- **HTML5**: Semantic markup and structure
- **CSS3**: Modern styling with animations and transitions
- **JavaScript ES6+**: Interactive features and language switching
- **Ionicons**: Beautiful icon library
- **Google Fonts**: Poppins font family

### Blog Stack
- **Jekyll**: Static site generator
- **Kramdown**: Markdown parser
- **Rouge**: Code syntax highlighting
- **MathJax**: LaTeX math formula rendering
- **Mermaid**: Diagram rendering engine
- **GitHub Pages**: Website hosting platform

## 🌐 Language Support

The website supports two languages:
- **English** (Default)
- **中文** (Chinese)

Click the language toggle button (🌐) in the navigation bar to switch between languages.

## 📱 Website Sections

### Main Sections
- **Home**: About me and introduction
- **Resume**: Education, experience, and skills
- **Projects**: Showcase of my work and projects
- **Interests**: Personal interests and hobbies
- **Contact**: Get in touch with me

### Blog Sections
- **Blog Home**: Article list in reverse chronological order
- **Article Detail**: Full content of a single article
- **Category Browse**: Filter articles by category
- **Tag Cloud**: Filter articles by tags

## 🚀 Deployment

This project is designed to be deployed on GitHub Pages. To deploy:

1. Push the code to your GitHub repository
2. Enable GitHub Pages in repository settings
3. Select the main branch as source
4. Your site will be available at `https://username.github.io/repository-name`


## 🎨 Customization

### Website Customization

To customize this website for your own use:

1. **Personal Information**: Update the content in `index.html`
2. **Images**: Replace images in the `assets/images/` folder
3. **Colors**: Modify CSS variables in `assets/css/style.css`
4. **Languages**: Update translation objects in the script section

### Blog Customization

#### Writing New Articles

1. Create a new file in `_posts/` directory with the format: `YYYY-MM-DD-title.md`
2. Add YAML Front Matter at the top of the file:

```yaml
---
layout: blog-post
title: Article Title
date: 2025-12-26 10:00:00 +0800
series: Series Name (optional)
categories: [Category1, Category2]
tags: [Tag1, Tag2]
author: Author Name
excerpt: Article summary
---
```

3. Write the content in Markdown, supporting:
   - **Code Highlighting**: Use \`\`\`language code blocks
   - **Math Formulas**: Use `$...$` (inline) or `$$...$$` (block)
   - **Mermaid Diagrams**: Use \`\`\`mermaid code blocks

#### Blog Configuration

Edit `_config.yml` file to configure:
- Site title and description
- Blog post permalink format
- Code highlighting theme
- Plugins (jekyll-feed, jekyll-sitemap)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Hyacehila**   Portfolio: [https://hyacehila.github.io](https://hyacehila.github.io)

## 🙏 Acknowledgments

- Original portfolio template from [ivansaul/personal-portfolio](https://github.com/ivansaul/personal-portfolio)
- Blog functionality powered by Jekyll
- Code highlighting by [Rouge](https://github.com/rouge-ruby/rouge)
- Math formula rendering by [MathJax](https://www.mathjax.org/)
- Diagram rendering by [Mermaid](https://mermaid.js.org/)

---

⭐ If you like this project, consider giving it a star!
